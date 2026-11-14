using SkiaSharp;

namespace SemanticCut.Core;

/// <summary>Loads a still image the same way video frames are prepared: squash to 384x384, RGB, normalized to [-1,1], CHW.</summary>
public static class ImageInput
{
    public static readonly HashSet<string> Extensions = new(StringComparer.OrdinalIgnoreCase)
        { ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif" };

    public static bool IsImage(string path) => Extensions.Contains(Path.GetExtension(path));

    public static float[] LoadChw(string path)
    {
        int s = SigLipEmbedder.ImageSize;
        using var src = SKBitmap.Decode(path) ?? throw new InvalidDataException($"Can't decode image: {path}");
        using var dst = src.Resize(new SKImageInfo(s, s, SKColorType.Rgba8888, SKAlphaType.Opaque),
                new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear))
            ?? throw new InvalidDataException($"Can't resize image: {path}");

        var px = dst.GetPixelSpan(); // RGBA, row-major
        int plane = s * s;
        var o = new float[3 * plane];
        for (int i = 0; i < plane; i++)
        {
            o[i] = px[i * 4] / 127.5f - 1f;
            o[plane + i] = px[i * 4 + 1] / 127.5f - 1f;
            o[2 * plane + i] = px[i * 4 + 2] / 127.5f - 1f;
        }
        return o;
    }
}
