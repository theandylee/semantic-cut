using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using SemanticCut.Core;

// Always format numbers with '.' regardless of the system locale.
System.Globalization.CultureInfo.DefaultThreadCurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
System.Globalization.CultureInfo.CurrentCulture = System.Globalization.CultureInfo.InvariantCulture;

// Optional "--models <path>" anywhere on the command line. Default: "models" next to semcut.exe (or $SEMCUT_MODELS).
string? modelsOpt = null;
bool noDownload = false;
{
    var list = args.ToList();
    int mi = list.IndexOf("--models");
    if (mi >= 0 && mi + 1 < list.Count) { modelsOpt = list[mi + 1]; list.RemoveRange(mi, 2); }
    noDownload = list.Remove("--no-download");
    args = list.ToArray();
}

// Resolved lazily so commands that don't need the model still work without it. If the models are missing from the
// default location (next to the exe) they are downloaded first, unless --no-download is given.
string ModelDir() => ModelLocator.FindOrDownloadAsync(modelsOpt, !noDownload, new ConsoleDownloadProgress(), CancellationToken.None)
    .GetAwaiter().GetResult();

try
{
switch (args.FirstOrDefault())
{
    case "inspect":
        Console.WriteLine("Providers: " + string.Join(", ", OrtEnv.Instance().GetAvailableProviders()));
        foreach (var path in args.Skip(1))
        {
            // Metadata only: no graph optimization (ORT 1.26's fp16 layer-norm fusion crashes on these graphs).
            using var so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL };
            using var s = new InferenceSession(path, so);
            Console.WriteLine($"== {Path.GetFileName(path)}");
            foreach (var i in s.InputMetadata) Console.WriteLine($"  in  {i.Key}: {i.Value.ElementType} [{string.Join(",", i.Value.Dimensions)}]");
            foreach (var o in s.OutputMetadata) Console.WriteLine($"  out {o.Key}: {o.Value.ElementType} [{string.Join(",", o.Value.Dimensions)}]");
        }
        break;

    case "tokens": // semcut tokens "text"  -> token ids (for comparing with the HF reference)
    {
        using var e = new SigLipEmbedder(ModelDir(), loadText: true, useGpu: false);
        Console.WriteLine(string.Join(",", e.Tokenize(string.Join(' ', args.Skip(1)))));
        break;
    }

    case "bench": // semcut bench <video> [batch] [everySeconds] [maxFrames]
    {
        var video = args[1];
        int batch = args.Length > 2 ? int.Parse(args[2]) : 16;
        double every = args.Length > 3 ? double.Parse(args[3], System.Globalization.CultureInfo.InvariantCulture) : 2.0;
        int max = args.Length > 4 ? int.Parse(args[4]) : 256;
        using var e = new SigLipEmbedder(ModelDir(), loadText: false, useGpu: true);

        int plane = 3 * SigLipEmbedder.ImageSize * SigLipEmbedder.ImageSize;
        var pending = new List<float[]>();
        int total = 0;
        double decodeMs = 0, inferMs = 0;
        var sw = Stopwatch.StartNew();
        var all = Stopwatch.StartNew();

        void Flush()
        {
            if (pending.Count == 0) return;
            var flat = new float[pending.Count * plane];
            for (int i = 0; i < pending.Count; i++) Array.Copy(pending[i], 0, flat, i * plane, plane);
            sw.Restart();
            e.EmbedImages(flat, pending.Count);
            inferMs += sw.Elapsed.TotalMilliseconds;
            total += pending.Count; pending.Clear();
        }

        sw.Restart();
        foreach (var (t, chw, _) in FrameSampler.Sample(video, every))
        {
            decodeMs += sw.Elapsed.TotalMilliseconds;
            pending.Add(chw);
            if (pending.Count == batch) Flush();
            if (total + pending.Count >= max) break;
            sw.Restart();
        }
        Flush();
        Console.WriteLine($"frames={total} wall={all.Elapsed.TotalSeconds:F1}s  decode~{decodeMs / 1000:F1}s  infer={inferMs / 1000:F1}s  => {total / (inferMs / 1000):F1} img/s (infer only)");
        break;
    }

    case "segment": // semcut segment <video> [every] [threshold] [maxSec]
    {
        double every = args.Length > 2 ? double.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture) : 2.0;
        double thr = args.Length > 3 ? double.Parse(args[3], System.Globalization.CultureInfo.InvariantCulture) : 0.85;
        double maxSec = args.Length > 4 ? double.Parse(args[4], System.Globalization.CultureInfo.InvariantCulture) : 30;
        using var e = new SigLipEmbedder(ModelDir(), loadText: false, useGpu: true);
        double dur = VideoIndexer.ProbeDuration(args[1]);
        var frames = VideoIndexer.Run(e, args[1], every, dur, null, CancellationToken.None);
        var segs = Segmenter.Build(frames, every, thr, maxSec, true, dur);
        Console.WriteLine($"duration={dur:F1}s frames={frames.Count} segments={segs.Count}");
        foreach (var s in segs.Take(40)) Console.WriteLine($"  {s.Start,7:F1} - {s.End,7:F1}  ({s.FrameCount} frames)");
        break;
    }

    case "search": // semcut search <video> <every> "query" ["query2" ...]
    {
        double every = double.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture);
        using var e = new SigLipEmbedder(ModelDir(), loadText: false, useGpu: true);
        double dur = VideoIndexer.ProbeDuration(args[1]);
        var frames = VideoIndexer.Run(e, args[1], every, dur, null, CancellationToken.None);
        var segs = Segmenter.Build(frames, every, 0.85, 30, true, dur);
        foreach (var q in args.Skip(3))
        {
            var sc = SegmentSearch.Score(frames, segs, e.EmbedText(q));
            var top = sc.Select((x, i) => (x.score, i)).OrderByDescending(x => x.score).First();
            Console.WriteLine($"{q,-45} best={top.score:F4} (segment #{top.i + 1}), mean={sc.Average(x => x.score):F4}");
        }
        break;
    }

    case "index": // semcut index <folder> [every]  -> indexes new/changed videos into <folder>/semantic_cut.db
    {
        double every = args.Length > 2 ? double.Parse(args[2], System.Globalization.CultureInfo.InvariantCulture) : 2.0;
        using var lib = Library.Open(args[1]);
        using var e = new SigLipEmbedder(ModelDir(), useGpu: true);
        Console.WriteLine($"{lib.Videos.Count} videos, {lib.Videos.Count(v => v.Stale)} need indexing");
        foreach (var v in lib.Videos.Where(v => v.Stale))
        {
            var sw = Stopwatch.StartNew();
            lib.Index(v, e, every, null, CancellationToken.None);
            Console.WriteLine($"  {v.Rel}: {v.Frames.Count} frames in {sw.Elapsed.TotalSeconds:F1}s");
        }
        break;
    }

    case "find": // semcut find <folder> "query" [topN] [threshold]
    {
        int topN = args.Length > 3 ? int.Parse(args[3]) : 10;
        double thr = args.Length > 4 ? double.Parse(args[4], System.Globalization.CultureInfo.InvariantCulture) : 0.85;
        var sw = Stopwatch.StartNew();
        using var lib = Library.Open(args[1]);
        Console.WriteLine($"opened {lib.Videos.Count(v => v.Indexed)}/{lib.Videos.Count} indexed videos in {sw.Elapsed.TotalSeconds:F2}s");
        // The query is text, or the path of an image file (image search; the vision model runs on the GPU).
        bool isImage = File.Exists(args[2]) && ImageInput.IsImage(args[2]);
        using var e = new SigLipEmbedder(ModelDir(), useGpu: isImage);
        var q = isImage ? e.EmbedImageFile(args[2]) : e.EmbedText(args[2]);
        sw.Restart();
        Library.Resegment(lib.Videos, thr, 30, true);
        var (top, total) = Library.Search(lib.Videos, q, topN);
        Console.WriteLine($"segment+search over {total} segments: {sw.Elapsed.TotalMilliseconds:F0} ms");
        foreach (var h in top)
            Console.WriteLine($"  {h.Score:F3}  {h.Video.Rel}  {h.Segment.Start:F1}-{h.Segment.End:F1}s  (best frame @ {h.BestTime:F1}s)");
        break;
    }

    case "download": // semcut download [--models <path>]  -> fetch/complete the models (and cuDNN) at an explicit location
    {
        var root = modelsOpt ?? ModelLocator.DefaultRoot;
        if (ModelDownloader.IsComplete(root)) { Console.WriteLine($"Models are already complete in {root}"); break; }
        Console.Error.WriteLine($"Downloading to {root} (about {ModelDownloader.ApproxTotalBytes / 1_000_000_000.0:F1} GB)…");
        await ModelDownloader.EnsureAsync(root, new ConsoleDownloadProgress(), CancellationToken.None);
        Console.WriteLine($"Done: {root}");
        break;
    }

    default:
        Console.WriteLine("""
            usage: semcut <command> ...

            Library (folder index, stored in <folder>/semantic_cut.db):
              index <folder> [everySec=2]                    index new/changed videos
              find  <folder> "query|image.jpg" [topN=10] [threshold=0.85]   search all indexed videos by text or by image

            Single video (nothing saved):
              segment <video> [everySec=2] [threshold=0.85] [maxSec=30]   print segments
              search  <video> <everySec> "query" ["query2" ...]           score queries against one video

            Diagnostics:
              bench <video> [batch=16] [everySec=2] [maxFrames=256]   GPU throughput
              tokens <text>                                           tokenizer ids
              inspect <onnx>...                                       model inputs/outputs

            Models:
              download [--models <path>]   download the SigLIP models + cuDNN (~3 GB) to <path> (default: "models" next to semcut.exe)

            Options:
              --models <path>   where the SigLIP models live (default: "models" next to semcut.exe, or $SEMCUT_MODELS)
              --no-download     don't auto-download missing models into the default location; fail instead
            """);
        break;
}
}
catch (Exception ex) when (ex is IOException or InvalidOperationException)
{
    // Expected user-facing problems (missing model/folder, index built with another model...): one clean line, no stack trace.
    // ONNX Runtime's native logger can leave an unterminated line on stderr; start on a fresh line so the message is readable.
    if (ex is MissingPrerequisiteException) Console.Error.WriteLine();
    Console.Error.WriteLine("error: " + ex.Message);
    return 1;
}
return 0;

/// <summary>Prints model-download progress to stderr as a single updating line per item.</summary>
sealed class ConsoleDownloadProgress : IProgress<DownloadProgress>
{
    string _item = "";
    DateTime _t0 = DateTime.UtcNow;
    long _done0;

    public void Report(DownloadProgress p)
    {
        if (p.Item != _item) { if (_item != "") Console.Error.WriteLine(); _item = p.Item; _t0 = DateTime.UtcNow; _done0 = p.Done; }
        double secs = Math.Max((DateTime.UtcNow - _t0).TotalSeconds, 0.1);
        double mbps = (p.Done - _done0) / 1e6 / secs;
        string size = p.Total > 0 ? $"{p.Done / 1e6,7:F0} / {p.Total / 1e6:F0} MB  {100.0 * p.Done / p.Total,3:F0}%  {mbps,5:F1} MB/s" : "";
        Console.Error.Write($"\r[{p.Index}/{p.Count}] {p.Item,-36} {size}   ");
    }
}
