using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using NAudio.Wave;

namespace SemanticCut.Gui;

/// <summary>
/// Plays [start, end) of any video ffmpeg can read, in a WPF Image, looping. Video: ffmpeg decodes to raw BGRA frames that are
/// written into a WriteableBitmap. Audio: ffmpeg decodes to PCM that NAudio plays. Frames are paced against a clock that
/// starts together with the audio. Replaces MediaElement, which can't open many codecs/containers.
/// </summary>
public sealed class FfmpegPlayer : IDisposable
{
    const int Fps = 30;                      // output frame rate (source is converted by ffmpeg's fps filter)
    const int AudioRate = 48000, AudioBytesPerSec = AudioRate * 2 * 2; // s16le stereo
    const int MaxW = 800, MaxH = 450;        // decode size cap (never upscaled)

    [DllImport("winmm.dll")] static extern uint timeBeginPeriod(uint ms);
    [DllImport("winmm.dll")] static extern uint timeEndPeriod(uint ms);

    readonly Image _target;
    readonly Dispatcher _ui;
    readonly object _lock = new();
    CancellationTokenSource? _cts;
    Task? _task;
    bool _paused, _muted, _timerRaised;

    // per-run state (guarded by _lock)
    Stopwatch _clock = new();
    WaveOutEvent? _wo;
    bool _runStarted;
    double _start, _dur;

    /// <summary>Raised on the UI thread when ffmpeg can't decode the file (message explains why).</summary>
    public event Action<string>? Failed;

    public FfmpegPlayer(Image target)
    {
        _target = target;
        _ui = target.Dispatcher;
    }

    /// <summary>Current position in the file, in seconds (within the segment).</summary>
    public double Position
    {
        get { lock (_lock) return _start + Math.Min(_clock.Elapsed.TotalSeconds, _dur); }
    }

    public bool Paused
    {
        get => _paused;
        set
        {
            lock (_lock)
            {
                _paused = value;
                if (!_runStarted) return; // applied when the run starts
                if (value) { _clock.Stop(); _wo?.Pause(); }
                else { _clock.Start(); _wo?.Play(); }
            }
        }
    }

    public bool Muted
    {
        get => _muted;
        set { lock (_lock) { _muted = value; if (_wo != null) _wo.Volume = value ? 0f : 1f; } }
    }

    /// <summary>Starts (or restarts) looped playback of [start, end) of <paramref name="path"/>. srcW/srcH may be 0 if unknown.</summary>
    public void Play(string path, double start, double end, int srcW, int srcH, double offset = 0, bool keepPaused = false)
    {
        bool wasPaused = _paused;
        Stop();

        // Decode size: source size, shrunk to fit MaxW x MaxH (even numbers, never upscaled).
        if (srcW <= 0 || srcH <= 0) { srcW = 1280; srcH = 720; }
        double k = Math.Min(1.0, Math.Min((double)MaxW / srcW, (double)MaxH / srcH));
        int w = Math.Max(2, (int)Math.Round(srcW * k) / 2 * 2), h = Math.Max(2, (int)Math.Round(srcH * k) / 2 * 2);

        // The new bitmap is shown only once its first frame has been written (see RunOnce), so the previous
        // picture stays on screen while ffmpeg starts instead of flashing black.
        var bmp = new WriteableBitmap(w, h, 96, 96, PixelFormats.Bgra32, null);

        _paused = keepPaused && wasPaused;
        if (!_timerRaised) { timeBeginPeriod(1); _timerRaised = true; } // 1 ms sleep granularity for smooth frame pacing
        var cts = _cts = new CancellationTokenSource();
        double dur = Math.Max(0.5, end - start);
        offset = Math.Clamp(offset, 0, dur - 0.3);
        _task = Task.Factory.StartNew(() => Loop(path, start, dur, offset, w, h, bmp, cts.Token),
            cts.Token, TaskCreationOptions.LongRunning, TaskScheduler.Default);
    }

    public void Stop()
    {
        var cts = _cts; var task = _task;
        _cts = null; _task = null;
        if (cts == null) return;
        cts.Cancel();
        try { task?.Wait(1500); } catch { /* cancelled */ }
        cts.Dispose();
        lock (_lock) { _runStarted = false; _clock.Reset(); }
        if (_timerRaised) { timeEndPeriod(1); _timerRaised = false; }
    }

    public void Dispose() => Stop();

    // ---- playback loop ---------------------------------------------------
    void Loop(string path, double start, double dur, double offset, int w, int h, WriteableBitmap bmp, CancellationToken ct)
    {
        try
        {
            // The first pass may start part-way in (after a seek); every later pass starts at the segment start.
            while (!ct.IsCancellationRequested)
            {
                if (!RunOnce(path, start + offset, dur - offset, w, h, bmp, ct)) return; // false = fatal (already reported)
                offset = 0;
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex) { ReportFailure("Playback error: " + ex.Message); }
    }

    /// <returns>false if playback failed and shouldn't loop.</returns>
    bool RunOnce(string path, double start, double dur, int w, int h, WriteableBitmap bmp, CancellationToken ct)
    {
        int frameBytes = w * h * 4;
        var inv = CultureInfo.InvariantCulture;
        string ss = start.ToString("0.###", inv), tt = dur.ToString("0.###", inv);

        Process? vp = null, ap = null;
        WaveOutEvent? wo = null;
        var frames = new BlockingCollection<byte[]>(boundedCapacity: 6);
        var vErr = new ConcurrentQueue<string>();
        var audioDone = false;
        BufferedWaveProvider? bwp = null;

        lock (_lock) { _clock = new Stopwatch(); _runStarted = false; _start = start; _dur = dur; }

        try
        {
            vp = StartFfmpeg(["-v", "error", "-nostdin", "-hwaccel", "auto", "-ss", ss, "-t", tt, "-i", path,
                "-an", "-sn", "-vf", $"fps={Fps},scale={w}:{h}:flags=fast_bilinear", "-pix_fmt", "bgra", "-f", "rawvideo", "-"], vErr);
            ap = StartFfmpeg(["-v", "quiet", "-nostdin", "-ss", ss, "-t", tt, "-i", path,
                "-vn", "-sn", "-ac", "2", "-ar", AudioRate.ToString(inv), "-f", "s16le", "-"], null);

            // video reader: exact-size frames into a bounded queue (back-pressure keeps ffmpeg from racing ahead)
            var vStream = vp.StandardOutput.BaseStream;
            var vReader = Task.Run(() =>
            {
                try
                {
                    while (!ct.IsCancellationRequested)
                    {
                        var buf = ArrayPool<byte>.Shared.Rent(frameBytes);
                        if (ReadExactly(vStream, buf, frameBytes) < frameBytes) { ArrayPool<byte>.Shared.Return(buf); break; }
                        frames.Add(buf, ct);
                    }
                }
                catch (OperationCanceledException) { }
                catch (IOException) { }
                finally { frames.CompleteAdding(); }
            }, ct);

            // audio: PCM into an NAudio buffer; silent if the file has no audio track or there is no output device
            try
            {
                bwp = new BufferedWaveProvider(new WaveFormat(AudioRate, 16, 2))
                { BufferDuration = TimeSpan.FromSeconds(4), DiscardOnBufferOverflow = true, ReadFully = true };
                wo = new WaveOutEvent { DesiredLatency = 80 };
                wo.Init(bwp);
                wo.Volume = _muted ? 0f : 1f;
            }
            catch { wo?.Dispose(); wo = null; bwp = null; }

            var aStream = ap.StandardOutput.BaseStream;
            var aReader = Task.Run(() =>
            {
                try
                {
                    var buf = new byte[16384];
                    int n;
                    while (!ct.IsCancellationRequested && (n = aStream.Read(buf, 0, buf.Length)) > 0)
                    {
                        if (bwp == null) continue;
                        while (bwp.BufferedBytes > AudioBytesPerSec * 2 && !ct.IsCancellationRequested) Thread.Sleep(15);
                        bwp.AddSamples(buf, 0, n);
                    }
                }
                catch (IOException) { }
                finally { audioDone = true; }
            }, ct);

            // wait for the first frame; none within 10 s (or EOF with nothing) = ffmpeg can't decode this
            if (!frames.TryTake(out var first, 10_000, ct))
            {
                vp.WaitForExit(500);
                var err = string.Join(" ", vErr.TakeLast(3));
                ReportFailure("ffmpeg couldn't decode this file." + (err.Length > 0 ? " " + err : ""));
                return false;
            }

            // pre-buffer ~150 ms of audio so sound and picture start together
            var wait = Stopwatch.StartNew();
            while (bwp != null && !audioDone && bwp.BufferedBytes < AudioBytesPerSec * 0.15 && wait.ElapsedMilliseconds < 1500)
                Thread.Sleep(5);

            lock (_lock)
            {
                _wo = wo;
                _runStarted = true;
                if (!_paused) { _clock.Start(); wo?.Play(); }
            }

            Present(bmp, first, w, h);
            // Queued after the first frame's write, so the image switches straight from the old picture to the new one.
            _ui.BeginInvoke(DispatcherPriority.Render, () => { if (!ct.IsCancellationRequested && !ReferenceEquals(_target.Source, bmp)) _target.Source = bmp; });

            // present frames on schedule: frame i is due at i / Fps seconds on the clock
            int index = 1;
            foreach (var f in frames.GetConsumingEnumerable(ct))
            {
                double due = index / (double)Fps;
                while (!ct.IsCancellationRequested)
                {
                    double now; bool paused;
                    lock (_lock) { now = _clock.Elapsed.TotalSeconds; paused = _paused; }
                    if (!paused && now >= due) break;
                    Thread.Sleep(paused ? 20 : 1);
                }
                ct.ThrowIfCancellationRequested();

                double late; lock (_lock) late = _clock.Elapsed.TotalSeconds - due;
                if (late > 0.12) ArrayPool<byte>.Shared.Return(f); // behind: drop this frame to catch up
                else Present(bmp, f, w, h);
                index++;
            }

            // let the segment (and its audio) play out to its full length before looping
            while (!ct.IsCancellationRequested)
            {
                double now; bool paused;
                lock (_lock) { now = _clock.Elapsed.TotalSeconds; paused = _paused; }
                if (!paused && now >= dur) break;
                Thread.Sleep(10);
            }
            return true;
        }
        finally
        {
            lock (_lock) { _runStarted = false; _wo = null; }
            try { wo?.Stop(); wo?.Dispose(); } catch { }
            foreach (var p in new[] { vp, ap })
            {
                if (p == null) continue;
                try { if (!p.HasExited) p.Kill(true); } catch { }
                p.Dispose();
            }
        }
    }

    int _inflight; // frames posted to the UI thread but not yet drawn

    /// <summary>Copies a frame into the bitmap on the UI thread. If the UI is already 2 frames behind, drops this one instead.</summary>
    void Present(WriteableBitmap bmp, byte[] frame, int w, int h)
    {
        if (Volatile.Read(ref _inflight) >= 2) { ArrayPool<byte>.Shared.Return(frame); return; }
        Interlocked.Increment(ref _inflight);
        _ui.BeginInvoke(DispatcherPriority.Render, () =>
        {
            try { bmp.WritePixels(new Int32Rect(0, 0, w, h), frame, w * 4, 0); }
            finally { ArrayPool<byte>.Shared.Return(frame); Interlocked.Decrement(ref _inflight); }
        });
    }

    void ReportFailure(string message) => _ui.BeginInvoke(() => Failed?.Invoke(message));

    static int ReadExactly(Stream s, byte[] buf, int count)
    {
        int got = 0;
        while (got < count)
        {
            int n = s.Read(buf, got, count - got);
            if (n <= 0) break;
            got += n;
        }
        return got;
    }

    static Process StartFfmpeg(string[] args, ConcurrentQueue<string>? errors)
    {
        var psi = new ProcessStartInfo("ffmpeg")
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true,
        };
        foreach (var a in args) psi.ArgumentList.Add(a);
        var p = Process.Start(psi)!;
        p.ErrorDataReceived += (sender, e) =>
        {
            if (e.Data == null || errors == null) return;
            errors.Enqueue(e.Data);
            while (errors.Count > 20) errors.TryDequeue(out _);
        };
        p.BeginErrorReadLine();
        return p;
    }
}
