namespace SemanticCut.Core;

/// <summary>A required external component is missing. The message says what and where to get it; Url is the download page.</summary>
public sealed class MissingPrerequisiteException(string what, string url, string? hint = null)
    : InvalidOperationException($"{what}{(hint != null ? "\n" + hint : "")}\nDownload: {url}")
{
    public string Url { get; } = url;
}

/// <summary>
/// Checks for components we can't bundle. (A missing .NET runtime is reported by the exe launcher itself, before any of our
/// code runs, with a download link.)
/// </summary>
public static class Prerequisites
{
    public const string FfmpegUrl = "https://www.gyan.dev/ffmpeg/builds/";
    public const string CudaUrl = "https://developer.nvidia.com/cuda-12-8-0-download-archive";

    /// <summary>Looks next to the executable first, then on PATH.</summary>
    public static string? Find(string tool)
    {
        var file = tool + ".exe";
        var dirs = new[] { AppContext.BaseDirectory }
            .Concat((Environment.GetEnvironmentVariable("PATH") ?? "").Split(';', StringSplitOptions.RemoveEmptyEntries));
        foreach (var d in dirs)
        {
            try
            {
                var p = Path.Combine(d.Trim().Trim('"'), file);
                if (File.Exists(p)) return p;
            }
            catch { /* malformed PATH entry */ }
        }
        return null;
    }

    public static void RequireFfmpeg()
    {
        foreach (var tool in new[] { "ffmpeg", "ffprobe" })
            if (Find(tool) == null)
                throw new MissingPrerequisiteException(
                    $"'{tool}' was not found. ffmpeg is needed to read video files.",
                    FfmpegUrl,
                    "Install it (for example: winget install Gyan.FFmpeg) and make sure ffmpeg and ffprobe are on your PATH, " +
                    "or put them next to semcut.exe.");
    }
}
