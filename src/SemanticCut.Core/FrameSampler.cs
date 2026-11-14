using System.Diagnostics;

namespace SemanticCut.Core;

/// <summary>Decodes a video with the ffmpeg binary and yields 384x384 frames at a fixed rate, as normalized CHW floats.</summary>
public static class FrameSampler
{
    const int S = SigLipEmbedder.ImageSize;

    public const int ThumbSize = 128;

    /// <summary>Yields (timestampSeconds, chwFloats[3*384*384] in [-1,1], thumbRgb[128*128*3]).</summary>
    public static IEnumerable<(double t, float[] chw, byte[] thumb)> Sample(string video, double everySeconds = 2.0, bool hwDecode = true)
    {
        var psi = new ProcessStartInfo("ffmpeg")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (var a in new[] { "-v", "error", "-nostdin" }) psi.ArgumentList.Add(a);
        if (hwDecode) { psi.ArgumentList.Add("-hwaccel"); psi.ArgumentList.Add("auto"); }
        foreach (var a in new[]
        {
            "-i", video,
            "-vf", $"fps=1/{everySeconds.ToString(System.Globalization.CultureInfo.InvariantCulture)},scale={S}:{S}:flags=bilinear",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        }) psi.ArgumentList.Add(a);

        using var p = Process.Start(psi)!;
        p.ErrorDataReceived += (_, _) => { };
        _ = p.StandardError.ReadToEndAsync();
        var stream = p.StandardOutput.BaseStream;
        var buf = new byte[S * S * 3];
        try
        {
            for (int idx = 0; ; idx++)
            {
                int got = 0;
                while (got < buf.Length)
                {
                    int r = stream.Read(buf, got, buf.Length - got);
                    if (r == 0) break;
                    got += r;
                }
                if (got < buf.Length) yield break;
                yield return (idx * everySeconds, ToChw(buf), Thumb(buf));
            }
        }
        finally
        {
            // Abandoned early (cancel/error): don't leave ffmpeg blocked on a full pipe.
            try { if (!p.HasExited) p.Kill(true); } catch { }
        }
    }

    static byte[] Thumb(byte[] rgb)
    {
        const int T = ThumbSize, k = S / T; // 3x3 box average
        var o = new byte[T * T * 3];
        for (int y = 0; y < T; y++)
            for (int x = 0; x < T; x++)
                for (int c = 0; c < 3; c++)
                {
                    int sum = 0;
                    for (int dy = 0; dy < k; dy++)
                        for (int dx = 0; dx < k; dx++)
                            sum += rgb[((y * k + dy) * S + x * k + dx) * 3 + c];
                    o[(y * T + x) * 3 + c] = (byte)(sum / (k * k));
                }
        return o;
    }

    static float[] ToChw(byte[] rgb)
    {
        var o = new float[3 * S * S];
        int plane = S * S;
        for (int i = 0; i < plane; i++)
        {
            o[i] = rgb[i * 3] / 127.5f - 1f;
            o[plane + i] = rgb[i * 3 + 1] / 127.5f - 1f;
            o[2 * plane + i] = rgb[i * 3 + 2] / 127.5f - 1f;
        }
        return o;
    }
}
