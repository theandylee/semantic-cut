using System.IO.Compression;
using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;

namespace SemanticCut.Core;

/// <param name="Item">What is being fetched right now (file name / step).</param>
/// <param name="Done">Bytes of this item done (or 0 for non-byte steps).</param>
/// <param name="Total">Total bytes of this item (0 if unknown).</param>
/// <param name="Index">1-based index of the item among <paramref name="Count"/> items.</param>
public sealed record DownloadProgress(string Item, long Done, long Total, int Index, int Count);

/// <summary>
/// Downloads what the app needs into a "models" folder: the SigLIP 2 ONNX models + tokenizer (Hugging Face) and
/// cuDNN 9 (NVIDIA redistributable). Resumable, SHA-256 verified, and installed atomically.
/// </summary>
public static class ModelDownloader
{
    const string HfBase = "https://huggingface.co/onnx-community/siglip2-so400m-patch14-384-ONNX/resolve/main/";
    const string CudnnUrl = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.10.2.21_cuda12-archive.zip";
    const string CudnnSha256 = "c1a4567d822ebda7373fa1f19255dff4942302de741f830160b6c7d1fb31af23";
    const long CudnnSize = 683336095;
    public const long RequiredFreeBytes = 4L << 30; // model files + cuDNN zip + extraction headroom

    sealed record Item(string Url, string Rel, long Size, string Sha256);

    // Sizes/hashes come from the Hugging Face LFS metadata of the repo above.
    static readonly Item[] ModelItems =
    [
        new(HfBase + "tokenizer.model", "tokenizer.model", 4241003, "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"),
        new(HfBase + "onnx/vision_model_fp16.onnx", "onnx/vision_model_fp16.onnx", 857034174, "7eff9ceb868eceea1aed8d9c5fa3da358eec042abf3f2c8bced5d97de065f9f3"),
        new(HfBase + "onnx/text_model_fp16.onnx", "onnx/text_model_fp16.onnx", 1416135143, "21086ca63010a7f9a4310c362ff351f26499f64461c942a0a71732daaba226ac"),
    ];

    public static long ApproxTotalBytes => ModelItems.Sum(i => i.Size) + CudnnSize;

    static readonly HttpClient Http = CreateClient();
    static HttpClient CreateClient()
    {
        var c = new HttpClient(new SocketsHttpHandler { AllowAutoRedirect = true, ConnectTimeout = TimeSpan.FromSeconds(30) })
        { Timeout = Timeout.InfiniteTimeSpan };
        c.DefaultRequestHeaders.UserAgent.ParseAdd("semcut-model-downloader/1.0");
        return c;
    }

    // ---- what is present ------------------------------------------------
    public static string ModelDir(string root) => Path.Combine(root, ModelLocator.DirName);

    public static bool ModelComplete(string root) =>
        ModelItems.All(i => File.Exists(Path.Combine(ModelDir(root), i.Rel.Replace('/', Path.DirectorySeparatorChar)))
                            && new FileInfo(Path.Combine(ModelDir(root), i.Rel.Replace('/', Path.DirectorySeparatorChar))).Length == i.Size);

    /// <summary>cuDNN is present in root\cudnn, or already installed system-wide (on PATH).</summary>
    public static bool CudnnPresent(string root) =>
        File.Exists(Path.Combine(root, "cudnn", "cudnn64_9.dll")) && File.Exists(Path.Combine(root, "cudnn", "cudnn_engines_precompiled64_9.dll"))
        || OnPath("cudnn64_9.dll");

    public static bool IsComplete(string root) => ModelComplete(root) && CudnnPresent(root);

    static bool OnPath(string dll) =>
        (Environment.GetEnvironmentVariable("PATH") ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries)
            .Any(d => { try { return File.Exists(Path.Combine(d.Trim(), dll)); } catch { return false; } });

    // ---- download -------------------------------------------------------
    public static async Task EnsureAsync(string root, IProgress<DownloadProgress>? progress, CancellationToken ct)
    {
        Directory.CreateDirectory(root);
        bool needModel = !ModelComplete(root), needCudnn = !CudnnPresent(root);
        if (!needModel && !needCudnn) return;

        var drive = new DriveInfo(Path.GetPathRoot(Path.GetFullPath(root))!);
        if (drive.IsReady && drive.AvailableFreeSpace < RequiredFreeBytes)
            throw new IOException($"Not enough free disk space on {drive.Name}: need about {RequiredFreeBytes >> 30} GB, " +
                                  $"have {drive.AvailableFreeSpace / (1L << 30)} GB.");

        int count = (needModel ? ModelItems.Length : 0) + (needCudnn ? 1 : 0), idx = 0;

        if (needModel)
            foreach (var it in ModelItems)
            {
                idx++;
                var dest = Path.Combine(ModelDir(root), it.Rel.Replace('/', Path.DirectorySeparatorChar));
                if (File.Exists(dest) && new FileInfo(dest).Length == it.Size) continue;
                await DownloadFileAsync(it.Url, dest, it.Size, it.Sha256, it.Rel, idx, count, progress, ct);
            }

        if (needCudnn)
        {
            idx++;
            await InstallCudnnAsync(root, idx, count, progress, ct);
        }
    }

    static async Task InstallCudnnAsync(string root, int idx, int count, IProgress<DownloadProgress>? progress, CancellationToken ct)
    {
        var tmpDir = Path.Combine(root, "_download");
        Directory.CreateDirectory(tmpDir);
        var zip = Path.Combine(tmpDir, "cudnn.zip");
        await DownloadFileAsync(CudnnUrl, zip, CudnnSize, CudnnSha256, "cuDNN 9 (NVIDIA)", idx, count, progress, ct);

        progress?.Report(new DownloadProgress("cuDNN 9: extracting", 0, 0, idx, count));
        var staging = Path.Combine(root, "cudnn.tmp");
        if (Directory.Exists(staging)) Directory.Delete(staging, true);
        Directory.CreateDirectory(staging);
        using (var za = ZipFile.OpenRead(zip))
            foreach (var e in za.Entries.Where(e => e.FullName.Contains("/bin/") && e.Name.EndsWith(".dll", StringComparison.OrdinalIgnoreCase)))
            {
                ct.ThrowIfCancellationRequested();
                e.ExtractToFile(Path.Combine(staging, e.Name), true);
            }
        if (!File.Exists(Path.Combine(staging, "cudnn64_9.dll")))
            throw new InvalidDataException("The cuDNN archive did not contain the expected DLLs.");

        var final = Path.Combine(root, "cudnn");
        if (Directory.Exists(final)) Directory.Delete(final, true);
        Directory.Move(staging, final); // atomic: a half-extracted cudnn folder never exists under the final name
        try { Directory.Delete(tmpDir, true); } catch { }
    }

    /// <summary>Resumable download to dest (via dest.part), verified by size + SHA-256, then atomically renamed.</summary>
    static async Task DownloadFileAsync(string url, string dest, long size, string sha256, string label, int idx, int count,
        IProgress<DownloadProgress>? progress, CancellationToken ct)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
        var part = dest + ".part";
        if (File.Exists(dest) && new FileInfo(dest).Length == size && Sha(dest) == sha256) return; // e.g. leftover cuDNN zip

        long have = File.Exists(part) ? new FileInfo(part).Length : 0;
        if (have > size) { File.Delete(part); have = 0; }

        if (have < size)
        {
            using var req = new HttpRequestMessage(HttpMethod.Get, url);
            if (have > 0) req.Headers.Range = new RangeHeaderValue(have, null);
            using var resp = await Http.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct);
            if (have > 0 && resp.StatusCode != HttpStatusCode.PartialContent) have = 0; // server ignored Range: start over
            resp.EnsureSuccessStatusCode();

            await using var src = await resp.Content.ReadAsStreamAsync(ct);
            await using var fs = new FileStream(part, have > 0 ? FileMode.Append : FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20);
            var buf = new byte[256 * 1024];
            long done = have;
            var last = DateTime.MinValue;
            progress?.Report(new DownloadProgress(label, done, size, idx, count));
            int n;
            while ((n = await src.ReadAsync(buf, ct)) > 0)
            {
                await fs.WriteAsync(buf.AsMemory(0, n), ct);
                done += n;
                if (progress != null && (DateTime.UtcNow - last).TotalMilliseconds > 250)
                {
                    last = DateTime.UtcNow;
                    progress.Report(new DownloadProgress(label, done, size, idx, count));
                }
            }
            progress?.Report(new DownloadProgress(label, done, size, idx, count));
        }

        long got = new FileInfo(part).Length;
        if (got != size)
            throw new IOException($"{label}: incomplete download ({got} of {size} bytes). Run again to resume.");

        progress?.Report(new DownloadProgress(label + " (verifying)", size, size, idx, count));
        if (Sha(part) != sha256)
        {
            File.Delete(part);
            throw new InvalidDataException($"{label}: checksum mismatch. The partial file was deleted; run again to re-download.");
        }
        File.Move(part, dest, overwrite: true);
    }

    static string Sha(string path)
    {
        using var fs = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(fs)).ToLowerInvariant();
    }
}
