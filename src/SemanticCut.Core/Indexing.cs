using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Numerics.Tensors;

namespace SemanticCut.Core;

/// <summary>One sampled frame: timestamp, L2-normalized embedding, small RGB thumbnail (128x128x3, empty when stored in the DB).</summary>
public sealed record FrameRecord(double T, float[] Emb, byte[] Thumb);

/// <summary>A contiguous time range of visually similar frames.</summary>
public sealed record Segment(int FirstFrame, int LastFrame, double Start, double End, float[] Centroid)
{
    public int FrameCount => LastFrame - FirstFrame + 1;
    public double Duration => End - Start;
}

public sealed record VideoInfo(double Duration, int Width, int Height);

public static class ModelLocator
{
    public const string DirName = "siglip2-so400m-384";

    public const string EnvVar = "SEMCUT_MODELS";

    /// <summary>
    /// Finds the SigLIP model folder. Order: explicit path (e.g. --models), then the SEMCUT_MODELS environment variable,
    /// then "models" next to the executable. A path may be either the "models" root (containing siglip2-so400m-384 and cudnn)
    /// or the model folder itself.
    /// </summary>
    public static string Find(string? explicitPath = null)
    {
        // An explicitly requested location (--models / env var) must be valid: a typo should fail loudly, not silently
        // fall back to the default and use some other model.
        foreach (var (root, source) in new[] { (explicitPath, "--models"), (Environment.GetEnvironmentVariable(EnvVar), EnvVar) })
        {
            if (string.IsNullOrWhiteSpace(root)) continue;
            var found = Probe(Path.GetFullPath(root));
            if (found != null) return found;
            throw new DirectoryNotFoundException(
                $"{source} points to '{root}', but no '{DirName}' model was found there " +
                $"(expected '{Path.Combine(Path.GetFullPath(root), DirName, "onnx", "vision_model_fp16.onnx")}').");
        }

        var def = Path.Combine(AppContext.BaseDirectory, "models");
        return Probe(def) ?? throw new DirectoryNotFoundException(
            $"SigLIP model not found. Expected '{Path.Combine(def, DirName)}'.\n" +
            $"Put the '{DirName}' folder in 'models' next to the executable, pass --models <path>, or set {EnvVar}.");
    }

    /// <summary>Accepts either the models root (containing the model folder) or the model folder itself.</summary>
    static string? Probe(string path)
    {
        var nested = Path.Combine(path, DirName);
        if (IsModelDir(nested)) return nested;
        return IsModelDir(path) ? path : null;
    }

    static bool IsModelDir(string p) =>
        File.Exists(Path.Combine(p, "onnx", "vision_model_fp16.onnx"))
        && File.Exists(Path.Combine(p, "onnx", "text_model_fp16.onnx"))
        && File.Exists(Path.Combine(p, "tokenizer.model"));

    public static string DefaultRoot => Path.Combine(AppContext.BaseDirectory, "models");

    /// <summary>True when no explicit location is configured and the default "models" folder is missing something we can download.</summary>
    public static bool NeedsDownload(string? explicitPath = null) =>
        string.IsNullOrWhiteSpace(explicitPath)
        && string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(EnvVar))
        && !ModelDownloader.IsComplete(DefaultRoot);

    /// <summary>Like Find, but downloads the models into the default folder first when they are missing (never for explicit paths).</summary>
    public static async Task<string> FindOrDownloadAsync(string? explicitPath, bool allowDownload,
        IProgress<DownloadProgress>? progress, CancellationToken ct)
    {
        if (allowDownload && NeedsDownload(explicitPath))
            await ModelDownloader.EnsureAsync(DefaultRoot, progress, ct);
        return Find(explicitPath);
    }
}

public static class VideoIndexer
{
    public static double ProbeDuration(string video) => Probe(video).Duration;

    /// <summary>Duration and display resolution (rotation-aware) via ffprobe. Zeros when unknown.</summary>
    public static VideoInfo Probe(string video)
    {
        try
        {
            var psi = new ProcessStartInfo("ffprobe")
            {
                RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true,
            };
            foreach (var a in new[]
            {
                "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height:stream_side_data=rotation:format=duration",
                "-of", "json", video,
            }) psi.ArgumentList.Add(a);
            using var p = Process.Start(psi)!;
            var json = p.StandardOutput.ReadToEnd();
            p.WaitForExit();
            using var doc = System.Text.Json.JsonDocument.Parse(json);
            var root = doc.RootElement;

            double dur = 0;
            if (root.TryGetProperty("format", out var fmt) && fmt.TryGetProperty("duration", out var ds))
                double.TryParse(ds.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out dur);

            int w = 0, h = 0;
            if (root.TryGetProperty("streams", out var streams) && streams.GetArrayLength() > 0)
            {
                var st = streams[0];
                if (st.TryGetProperty("width", out var wv)) w = wv.GetInt32();
                if (st.TryGetProperty("height", out var hv)) h = hv.GetInt32();
                if (st.TryGetProperty("side_data_list", out var sd))
                    foreach (var item in sd.EnumerateArray())
                        if (item.TryGetProperty("rotation", out var rot) && rot.ValueKind == System.Text.Json.JsonValueKind.Number
                            && Math.Abs(rot.GetDouble()) is 90 or 270)
                            (w, h) = (h, w); // phone footage: stored landscape, displayed portrait
            }
            return new VideoInfo(dur, w, h);
        }
        catch { return new VideoInfo(0, 0, 0); }
    }

    /// <summary>
    /// Decodes + embeds every frame at the given interval. Decoding/preprocessing runs on a producer thread so it overlaps
    /// with GPU inference. Frame data is returned so segmentation can be re-run cheaply.
    /// </summary>
    public static List<FrameRecord> Run(SigLipEmbedder emb, string video, double every, double duration,
        IProgress<(int done, int total)>? progress, CancellationToken ct, int batch = 8)
    {
        Prerequisites.RequireFfmpeg();
        int total = duration > 0 ? (int)Math.Ceiling(duration / every) : 0;
        int plane = 3 * SigLipEmbedder.ImageSize * SigLipEmbedder.ImageSize;
        var result = new List<FrameRecord>();
        var pend = new List<(double t, float[] chw, byte[] thumb)>();

        void Flush()
        {
            if (pend.Count == 0) return;
            var flat = new float[pend.Count * plane];
            for (int i = 0; i < pend.Count; i++) Array.Copy(pend[i].chw, 0, flat, i * plane, plane);
            var embs = emb.EmbedImages(flat, pend.Count);
            for (int i = 0; i < pend.Count; i++) result.Add(new FrameRecord(pend[i].t, embs[i], pend[i].thumb));
            pend.Clear();
            progress?.Report((result.Count, Math.Max(total, result.Count)));
        }

        using var queue = new BlockingCollection<(double, float[], byte[])>(boundedCapacity: 24);
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(ct);
        Exception? producerError = null;
        var producer = Task.Run(() =>
        {
            try { foreach (var f in FrameSampler.Sample(video, every)) queue.Add(f, stop.Token); }
            catch (OperationCanceledException) { }
            catch (Exception e) { producerError = e; }
            finally { queue.CompleteAdding(); }
        });

        try
        {
            foreach (var f in queue.GetConsumingEnumerable(ct))
            {
                pend.Add(f);
                if (pend.Count == batch) Flush();
            }
            Flush();
        }
        finally
        {
            stop.Cancel(); // unblocks the producer (and kills ffmpeg via the sampler's finally) on cancel/error
            producer.Wait();
        }
        if (producerError != null) throw new InvalidOperationException("Decoding failed: " + producerError.Message, producerError);
        return result;
    }
}

public static class SegmentSearch
{
    /// <summary>Scores each segment by its best-matching frame (cosine vs. a normalized query embedding).</summary>
    public static (float score, int bestFrame)[] Score(IReadOnlyList<FrameRecord> frames, IReadOnlyList<Segment> segs, float[] query)
    {
        var res = new (float, int)[segs.Count];
        for (int s = 0; s < segs.Count; s++)
        {
            float best = float.MinValue; int bi = segs[s].FirstFrame;
            for (int i = segs[s].FirstFrame; i <= segs[s].LastFrame; i++)
            {
                float d = TensorPrimitives.Dot(query, frames[i].Emb);
                if (d > best) { best = d; bi = i; }
            }
            res[s] = (best, bi);
        }
        return res;
    }
}

public static class Segmenter
{
    /// <summary>
    /// Greedy segmentation: a frame joins the current segment while its cosine similarity to the segment's running centroid
    /// stays >= threshold and the segment stays under maxSeconds. With ignoreBlips, a single outlier frame followed by a
    /// frame that matches again is absorbed instead of splitting. Allocation-free per frame (running sum + squared norm).
    /// </summary>
    public static List<Segment> Build(IReadOnlyList<FrameRecord> f, double every, double threshold, double maxSeconds,
        bool ignoreBlips, double duration)
    {
        var res = new List<Segment>();
        if (f.Count == 0) return res;

        int first = 0;
        var sum = (float[])f[0].Emb.Clone();
        double sumSq = TensorPrimitives.Dot(sum, sum);

        double EndOf(int last) => duration > 0 ? Math.Min(f[last].T + every, duration) : f[last].T + every;
        void Close(int last) => res.Add(new Segment(first, last, f[first].T, EndOf(last), Normalize(sum)));

        for (int i = 1; i < f.Count; i++)
        {
            double norm = Math.Sqrt(sumSq);
            float dot = TensorPrimitives.Dot(sum, f[i].Emb);
            double sim = dot / norm;
            bool tooLong = f[i].T + every - f[first].T > maxSeconds;
            bool diverges = sim < threshold;

            if (diverges && !tooLong && ignoreBlips && i + 1 < f.Count
                && TensorPrimitives.Dot(sum, f[i + 1].Emb) / norm >= threshold)
                continue; // blip: stays in the segment's range but doesn't pull the centroid

            if (diverges || tooLong)
            {
                Close(i - 1);
                first = i;
                sum = (float[])f[i].Emb.Clone();
                sumSq = TensorPrimitives.Dot(sum, sum);
            }
            else
            {
                sumSq += 2.0 * dot + TensorPrimitives.Dot(f[i].Emb, f[i].Emb);
                TensorPrimitives.Add(sum, f[i].Emb, sum);
            }
        }
        Close(f.Count - 1);
        return res;
    }

    static float[] Normalize(float[] v)
    {
        double s = TensorPrimitives.Dot(v, v);
        float inv = s > 0 ? (float)(1 / Math.Sqrt(s)) : 0;
        var o = new float[v.Length];
        TensorPrimitives.Multiply(v, inv, o);
        return o;
    }
}
