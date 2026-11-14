using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

namespace SemanticCut.Core;

/// <summary>SigLIP 2 (so400m patch14-384) image + text embedder over ONNX Runtime.</summary>
public sealed class SigLipEmbedder : IDisposable
{
    public const int ImageSize = 384;
    public const int Dim = 1152;
    public const int MaxTokens = 64;
    public const string ModelId = "siglip2-so400m-patch14-384-fp16";

    readonly string _dir;
    readonly object _textLock = new();
    readonly bool _gpu;
    InferenceSession? _vision;
    InferenceSession? _text;
    Tokenizer? _tok;
    const int _padId = 0; // Gemma <pad>
    const int _eosId = 1; // <eos>

    /// <summary>Both towers load lazily on first use (vision ~0.9 GB on GPU, text ~1.4 GB on CPU); loadText forces the text tower now.</summary>
    public SigLipEmbedder(string modelDir, bool loadText = false, bool useGpu = true)
    {
        _dir = modelDir;
        _gpu = useGpu;
        CudaSetup.Configure(modelDir);
        if (loadText) EnsureText();
    }

    public bool TextLoaded => _text != null;

    InferenceSession EnsureVision()
    {
        lock (_textLock)
            return _vision ??= new InferenceSession(Path.Combine(_dir, "onnx", "vision_model_fp16.onnx"), Options(_gpu));
    }

    public void EnsureText()
    {
        lock (_textLock)
        {
            if (_text != null) return;
            // Text tower is one 64-token query at a time: CPU is plenty and saves VRAM.
            // BASIC level: ORT 1.26's SimplifiedLayerNormFusion crashes on this fp16 graph at higher levels.
            var text = new InferenceSession(Path.Combine(_dir, "onnx", "text_model_fp16.onnx"),
                Options(false, GraphOptimizationLevel.ORT_ENABLE_BASIC));
            using var fs = File.OpenRead(Path.Combine(_dir, "tokenizer.model"));
            _tok = LlamaTokenizer.Create(fs, addBeginOfSentence: false, addEndOfSentence: false);
            _text = text;
        }
    }

    static SessionOptions Options(bool gpu, GraphOptimizationLevel level = GraphOptimizationLevel.ORT_ENABLE_ALL)
    {
        var o = new SessionOptions { GraphOptimizationLevel = level };
        if (gpu)
        {
            try { o.AppendExecutionProvider_CUDA(0); }
            catch (OnnxRuntimeException ex)
            {
                var reason = ex.Message.Split('\n').FirstOrDefault(l => l.Contains("which depends on") || l.Contains("Error loading")) ?? ex.Message.Split('\n')[0];
                throw new MissingPrerequisiteException(
                    "The NVIDIA GPU runtime (CUDA 12.x) could not be loaded.",
                    Prerequisites.CudaUrl,
                    $"You need an NVIDIA GPU with a current driver and the CUDA Toolkit 12.x installed.\nDetails: {reason.Trim()}");
            }
        }
        return o;
    }

    public IReadOnlyList<int> Tokenize(string text)
    {
        // SigLIP was trained on lowercase text, padded to max_length=64 with <pad>.
        EnsureText();
        var ids = _tok!.EncodeToIds(text.ToLowerInvariant()).ToList();
        if (ids.Count > MaxTokens - 1) ids = ids.Take(MaxTokens - 1).ToList();
        ids.Add(_eosId);
        while (ids.Count < MaxTokens) ids.Add(_padId);
        return ids;
    }

    /// <summary>pixels: N * 3 * 384 * 384 floats, CHW, already normalized to [-1,1]. Returns N L2-normalized vectors.</summary>
    public float[][] EmbedImages(float[] pixels, int n)
    {
        var t = new DenseTensor<float>(pixels, [n, 3, ImageSize, ImageSize]);
        using var res = EnsureVision().Run([NamedOnnxValue.CreateFromTensor("pixel_values", t)], ["pooler_output"]);
        return Normalize(res[0].AsTensor<float>().ToArray(), n);
    }

    /// <summary>Embeds a still image file (jpg/png/bmp/webp/gif) into the same space as video frames and text.</summary>
    public float[] EmbedImageFile(string path) => EmbedImages(ImageInput.LoadChw(path), 1)[0];

    public float[] EmbedText(string text)
    {
        EnsureText();
        var ids = Tokenize(text).Select(i => (long)i).ToArray();
        var t = new DenseTensor<long>(ids, [1, MaxTokens]);
        using var res = _text!.Run([NamedOnnxValue.CreateFromTensor("input_ids", t)], ["pooler_output"]);
        return Normalize(res[0].AsTensor<float>().ToArray(), 1)[0];
    }

    static float[][] Normalize(float[] flat, int n)
    {
        var o = new float[n][];
        for (int i = 0; i < n; i++)
        {
            var v = new float[Dim];
            Array.Copy(flat, i * Dim, v, 0, Dim);
            double s = 0; foreach (var x in v) s += (double)x * x;
            float inv = (float)(1.0 / Math.Sqrt(s));
            for (int j = 0; j < Dim; j++) v[j] *= inv;
            o[i] = v;
        }
        return o;
    }

    public static float Cosine(float[] a, float[] b)
    {
        float d = 0; for (int i = 0; i < a.Length; i++) d += a[i] * b[i]; return d;
    }

    public void Dispose() { _vision?.Dispose(); _text?.Dispose(); }
}

/// <summary>Makes the locally-unpacked cuDNN (models/cudnn) and CUDA toolkit DLLs discoverable.</summary>
static class CudaSetup
{
    static bool _done;
    public static void Configure(string modelDir)
    {
        if (_done) return; _done = true;
        var extra = new List<string>();
        var cudnn = Path.GetFullPath(Path.Combine(modelDir, "..", "cudnn"));
        if (Directory.Exists(cudnn)) extra.Add(cudnn);
        var cuda = Environment.GetEnvironmentVariable("CUDA_PATH_V12_8") ?? Environment.GetEnvironmentVariable("CUDA_PATH");
        if (cuda != null && Directory.Exists(Path.Combine(cuda, "bin"))) extra.Add(Path.Combine(cuda, "bin"));
        var path = Environment.GetEnvironmentVariable("PATH") ?? "";
        Environment.SetEnvironmentVariable("PATH", string.Join(';', extra) + ";" + path);
    }
}
