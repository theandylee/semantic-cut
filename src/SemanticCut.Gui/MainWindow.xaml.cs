using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Threading;
using Microsoft.Win32;
using SemanticCut.Core;

namespace SemanticCut.Gui;

public static class Fmts
{
    public static string Duration(double sec)
    {
        var t = TimeSpan.FromSeconds(Math.Round(sec));
        int h = (int)t.TotalHours;
        return h > 0 ? $"{h}:{t.Minutes:00}:{t.Seconds:00}" : $"{t.Minutes}:{t.Seconds:00}";
    }

    public static string Size(long bytes) => bytes switch
    {
        >= 1L << 30 => $"{bytes / (double)(1L << 30):0.##} GB",
        >= 1L << 20 => $"{bytes / (double)(1L << 20):0.#} MB",
        _ => $"{bytes / 1024.0:0} KB",
    };
}

/// <summary>Video-list row; reads live from the LibraryVideo so background probing can update it in place.</summary>
public sealed class VideoVm(LibraryVideo v) : System.ComponentModel.INotifyPropertyChanged
{
    public event System.ComponentModel.PropertyChangedEventHandler? PropertyChanged;
    public string Name => v.Rel;
    public string Meta => string.Join("  ·  ",
        v.Width > 0 ? $"{v.Width}×{v.Height}" : "…",
        v.Duration > 0 ? Fmts.Duration(v.Duration) : "…",
        Fmts.Size(v.Size));
    public string Status => v.Stale ? "not indexed" : $"indexed · {v.Frames.Count} frames · every {v.Every:0.#} s";
    public void Notify() => PropertyChanged?.Invoke(this, new(null)); // null = all properties
}

/// <summary>List row for one segment. Thumbnails are produced lazily (only for rows the list actually realizes).</summary>
public sealed class SegmentVm(string title, string info, Func<IReadOnlyList<BitmapSource>> makeThumbs)
{
    IReadOnlyList<BitmapSource>? _thumbs;
    public string Title { get; } = title;
    public string Info { get; } = info;
    public IReadOnlyList<BitmapSource> Thumbs => _thumbs ??= makeThumbs();
}

/// <summary>One row of the segment list: a segment of some video, optionally with a search score.</summary>
public sealed record ViewItem(LibraryVideo V, int Seg, float Score, int Best, int Rank);

public partial class MainWindow : Window
{
    const int TopN = 50;  // results shown for a library-wide search
    const int TopK = 10;  // results kept fully bright on the timeline

    static readonly Color[] Palette =
    [
        Color.FromRgb(0x3b, 0x82, 0xf6), Color.FromRgb(0x10, 0xb9, 0x81), Color.FromRgb(0xf5, 0x9e, 0x0b),
        Color.FromRgb(0xef, 0x44, 0x44), Color.FromRgb(0x8b, 0x5c, 0xf6), Color.FromRgb(0x14, 0xb8, 0xa6),
    ];

    SigLipEmbedder? _emb;
    Library? _lib;                       // null in single-video (no DB) mode
    List<LibraryVideo> _videos = [];
    LibraryVideo? _cur;                  // video shown on the timeline
    List<ViewItem> _view = [];           // rows of the segment list
    int _activeSeg = -1;                 // segment of _cur currently selected/playing
    CancellationTokenSource? _cts, _probeCts;
    List<VideoVm> _videoVms = [];
    readonly List<Rectangle> _rects = [];
    bool _running, _suppress;

    float[]? _query;
    string _queryText = "";
    float[]? _imageEmb;                  // embedding of the query image, if one is set
    string _imageName = "";

    // player
    readonly DispatcherTimer _timer = new() { Interval = TimeSpan.FromMilliseconds(100) };
    readonly FfmpegPlayer _player;
    Segment? _playing;

    public MainWindow()
    {
        InitializeComponent();
        UpdateLabels();
        // Slider marks its own press as handled, so a plain XAML handler would never run: ask for handled events too.
        SeekBar.AddHandler(UIElement.PreviewMouseLeftButtonDownEvent, new MouseButtonEventHandler(SeekBar_MouseDown), true);
        SeekBar.AddHandler(UIElement.MouseLeftButtonUpEvent, new MouseButtonEventHandler(SeekBar_MouseUp), true);
        _player = new FfmpegPlayer(PlayerImage);
        _player.Failed += Player_Failed;
        _timer.Tick += Timer_Tick;
        Loaded += async (_, _) =>
        {
            // Optional startup argument: a folder (opens its index) or a video file.
            var a = Environment.GetCommandLineArgs();
            if (a.Length > 1)
            {
                if (Directory.Exists(a[1])) await LoadFolderAsync(a[1]);
                else if (File.Exists(a[1])) SetVideo(a[1]);
            }
        };
    }

    sealed class ModelsUnavailableException(string message) : Exception(message);

    /// <summary>A required external component (ffmpeg, CUDA runtime) is missing: say so, and offer to open its download page.</summary>
    void HandleMissing(MissingPrerequisiteException ex)
    {
        StatusText.Text = ex.Message.Split('\n')[0];
        var r = MessageBox.Show(this, ex.Message + "\n\nOpen the download page now?", "A required component is missing",
            MessageBoxButton.YesNo, MessageBoxImage.Warning);
        if (r != MessageBoxResult.Yes) return;
        try { Process.Start(new ProcessStartInfo(ex.Url) { UseShellExecute = true }); }
        catch (Exception e) { MessageBox.Show(this, "Couldn't open the browser: " + e.Message + "\n" + ex.Url); }
    }

    /// <summary>
    /// Returns the shared embedder. If the models are missing from the default location (next to the exe), asks first and
    /// then downloads them (resumable, checksum-verified) with progress in the status bar.
    /// </summary>
    async Task<SigLipEmbedder> GetEmbedderAsync(CancellationToken ct = default)
    {
        if (_emb != null) return _emb;

        if (ModelLocator.NeedsDownload())
        {
            var answer = MessageBox.Show(this,
                $"The SigLIP models are not installed yet.\n\nDownload them now? About {ModelDownloader.ApproxTotalBytes / 1e9:F1} GB " +
                $"(the model from Hugging Face, cuDNN from NVIDIA) into:\n{ModelLocator.DefaultRoot}\n\nAn interrupted download resumes next time.",
                "Download models?", MessageBoxButton.YesNo, MessageBoxImage.Question);
            if (answer != MessageBoxResult.Yes)
                throw new ModelsUnavailableException("The models are missing and the download was declined.");

            string item = ""; var t0 = DateTime.UtcNow; long done0 = 0;
            var progress = new Progress<DownloadProgress>(p =>
            {
                if (p.Item != item) { item = p.Item; t0 = DateTime.UtcNow; done0 = p.Done; }
                double mbps = (p.Done - done0) / 1e6 / Math.Max((DateTime.UtcNow - t0).TotalSeconds, 0.1);
                Progress.Value = p.Total > 0 ? (double)p.Done / p.Total : 0;
                StatusText.Text = $"Downloading models [{p.Index}/{p.Count}] {p.Item}: {p.Done / 1e6:F0} / {p.Total / 1e6:F0} MB ({mbps:F1} MB/s)";
            });
            await ModelDownloader.EnsureAsync(ModelLocator.DefaultRoot, progress, ct);
            Progress.Value = 0;
        }

        StatusText.Text = "Loading model (first run takes a few seconds)…";
        var dir = ModelLocator.Find();
        return _emb = await Task.Run(() => new SigLipEmbedder(dir, useGpu: true));
    }

    double MaxSeconds => NoLimitBox.IsChecked == true ? double.PositiveInfinity : MaxSlider.Value;
    double EverySeconds => double.Parse(((ComboBoxItem)EveryBox.SelectedItem).Content.ToString()!, CultureInfo.InvariantCulture);
    bool AnyIndexed => _videos.Any(v => v.Indexed);

    // ---- opening things -------------------------------------------------
    void Open_Click(object s, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "Video|*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.mts;*.mxf|All files|*.*" };
        if (dlg.ShowDialog() == true) SetVideo(dlg.FileName);
    }

    async void OpenFolder_Click(object s, RoutedEventArgs e)
    {
        var dlg = new OpenFolderDialog { Title = "Choose a folder of footage" };
        if (dlg.ShowDialog() == true) await LoadFolderAsync(dlg.FolderName);
    }

    // ---- recent folders / files ------------------------------------------
    void Recent_Click(object s, RoutedEventArgs e)
    {
        var menu = new ContextMenu
        {
            PlacementTarget = RecentBtn,
            Placement = System.Windows.Controls.Primitives.PlacementMode.Bottom,
            MaxWidth = 640,
        };
        var recent = RecentStore.Load();
        if (recent.Count == 0) menu.Items.Add(new MenuItem { Header = "(nothing opened yet)", IsEnabled = false });

        foreach (var path in recent)
        {
            bool isDir = Directory.Exists(path);
            var item = new MenuItem
            {
                // A TextBlock header: a plain string would treat '_' in names as an access-key marker.
                Header = new TextBlock { Text = path, TextTrimming = TextTrimming.CharacterEllipsis },
                Icon = new TextBlock { Text = isDir ? "" : "", FontFamily = new FontFamily("Segoe MDL2 Assets") }, // folder / video
                ToolTip = path,
            };
            var p = path;
            item.Click += async (_, _) => await OpenRecentAsync(p);
            menu.Items.Add(item);
        }

        if (recent.Count > 0)
        {
            menu.Items.Add(new Separator());
            var clear = new MenuItem { Header = "Clear list" };
            clear.Click += (_, _) => RecentStore.Clear();
            menu.Items.Add(clear);
        }
        menu.IsOpen = true;
    }

    async Task OpenRecentAsync(string path)
    {
        if (_running) return;
        if (Directory.Exists(path)) await LoadFolderAsync(path);
        else if (File.Exists(path)) SetVideo(path);
        else
        {
            RecentStore.Remove(path);
            StatusText.Text = "No longer exists, removed from the list: " + path;
        }
    }

    async void Window_Drop(object s, DragEventArgs e)
    {
        if (e.Data.GetData(DataFormats.FileDrop) is not string[] f || f.Length == 0) return;
        if (Directory.Exists(f[0])) await LoadFolderAsync(f[0]);
        else if (ImageInput.IsImage(f[0])) await SetQueryImageAsync(f[0]); // dropped image = image query
        else SetVideo(f[0]);
    }

    void ResetState()
    {
        StopPlayback();
        _probeCts?.Cancel();
        _lib?.Dispose(); _lib = null;
        _videos = []; _videoVms = []; _cur = null; _view = []; _activeSeg = -1; _query = null; _queryText = "";
        _imageEmb = null; QueryThumb.Visibility = Visibility.Collapsed;
        SearchBox.Text = ""; TotalText.Text = "";
        SegList.ItemsSource = null; VideoList.ItemsSource = null;
        Timeline.Children.Clear(); _rects.Clear();
        Progress.Value = 0;
    }

    void SetVideo(string path)
    {
        if (_running) return;
        ResetState();
        var v = new LibraryVideo { Rel = System.IO.Path.GetFileName(path), FullPath = path };
        _videos = [v]; _cur = v;
        VideoPanel.Visibility = Visibility.Collapsed;
        PathBox.Text = path;
        RecentStore.Add(path);
        StatusText.Text = "Ready. Press Process. (Single video: results are not saved. Open a folder to keep an index.)";
        UpdateButtons();
    }

    async Task LoadFolderAsync(string folder)
    {
        if (_running) return;
        ResetState();
        PathBox.Text = folder;
        StatusText.Text = "Scanning folder and loading the index…";
        SetBusyUi(true);
        try
        {
            var sw = Stopwatch.StartNew();
            _lib = await Task.Run(() => Library.Open(folder));
            _videos = _lib.Videos;
            RecentStore.Add(folder);
            _cur = _videos.FirstOrDefault(v => v.Indexed) ?? _videos.FirstOrDefault();
            VideoPanel.Visibility = Visibility.Visible;
            RefreshVideoList();
            StartProbing(_lib);
            if (AnyIndexed) Resegment(); else StatusText.Text = "";
            int stale = _videos.Count(v => v.Stale);
            StatusText.Text = $"{_videos.Count} videos found, {_videos.Count - stale} already indexed" +
                              (stale > 0 ? $", {stale} new/changed. Press “Process folder”." : ". Everything is up to date.") +
                              $"  (loaded in {sw.Elapsed.TotalSeconds:F1}s)";
        }
        catch (Exception ex)
        {
            _lib?.Dispose(); _lib = null; _videos = [];
            StatusText.Text = "Couldn't open folder: " + ex.Message;
            MessageBox.Show(ex.Message, "Open folder");
        }
        finally { SetBusyUi(false); UpdateButtons(); }
    }

    // ---- processing -----------------------------------------------------
    async void Run_Click(object s, RoutedEventArgs e)
    {
        if (_running) { _cts?.Cancel(); return; }
        var todo = _lib == null ? _videos.ToList() : _videos.Where(v => v.Stale).ToList();
        if (todo.Count == 0) return;

        double every = EverySeconds;
        _running = true;
        _cts = new CancellationTokenSource();
        var ct = _cts.Token;
        SetBusyUi(true);
        StopPlayback();
        var started = DateTime.Now;
        int done = 0;

        try
        {
            Prerequisites.RequireFfmpeg(); // check first, before offering a 3 GB model download
            var emb = await GetEmbedderAsync(ct);

            for (int k = 0; k < todo.Count; k++)
            {
                ct.ThrowIfCancellationRequested();
                var v = todo[k]; int num = k + 1;
                var t0 = DateTime.Now;
                Progress.Value = 0;
                var progress = new Progress<(int done, int total)>(p =>
                {
                    Progress.Value = p.total > 0 ? (double)p.done / p.total : 0;
                    double fps = p.done / Math.Max((DateTime.Now - t0).TotalSeconds, 0.1);
                    StatusText.Text = $"[{num}/{todo.Count}] {v.Rel}: frames {p.done}/{p.total}  ({fps:F1} frames/s)";
                });

                if (_lib != null) await Task.Run(() => _lib.Index(v, emb, every, progress, ct));
                else await Task.Run(() => IndexAdHoc(v, emb, every, progress, ct));
                done++;
                RefreshVideoList();
            }
            Progress.Value = 1;
            Resegment();
            StatusText.Text = $"Done in {(DateTime.Now - started).TotalSeconds:F0}s: {done} video(s) processed. " + StatusText.Text;
        }
        catch (OperationCanceledException)
        {
            if (AnyIndexed) Resegment();
            StatusText.Text = $"Cancelled. {done} video(s) finished and saved.";
        }
        catch (ModelsUnavailableException ex)
        {
            StatusText.Text = ex.Message;
        }
        catch (MissingPrerequisiteException ex)
        {
            HandleMissing(ex);
        }
        catch (Exception ex)
        {
            StatusText.Text = "Error: " + ex.Message;
            // Expected problems (disk full, network, bad checksum) get a plain message; anything else gets the full trace.
            MessageBox.Show(ex is IOException or System.Net.Http.HttpRequestException or InvalidDataException ? ex.Message : ex.ToString(), "Processing failed");
        }
        finally
        {
            _running = false;
            SetBusyUi(false);
            RefreshVideoList();
            UpdateButtons();
        }
    }

    static void IndexAdHoc(LibraryVideo v, SigLipEmbedder emb, double every, IProgress<(int, int)> progress, CancellationToken ct)
    {
        double dur = VideoIndexer.ProbeDuration(v.FullPath);
        var frames = VideoIndexer.Run(emb, v.FullPath, every, dur, progress, ct);
        if (dur <= 0 && frames.Count > 0) dur = frames[^1].T + every;
        v.Frames = frames; v.Duration = dur; v.Every = every; v.Segments = []; v.Stale = false;
    }

    void SetBusyUi(bool busy)
    {
        OpenBtn.IsEnabled = OpenFolderBtn.IsEnabled = RecentBtn.IsEnabled = !busy;
        EveryBox.IsEnabled = VideoList.IsEnabled = !busy;
        if (busy) { SetSearchEnabled(false); ClearBtn.IsEnabled = false; }
        RunBtn.Content = _running ? "Cancel" : RunBtn.Content;
        RunBtn.IsEnabled = _running || RunBtn.IsEnabled;
    }

    void UpdateButtons()
    {
        int stale = _videos.Count(v => v.Stale);
        if (_lib == null) { RunBtn.Content = "Process"; RunBtn.IsEnabled = _videos.Count > 0; }
        else
        {
            RunBtn.Content = stale > 0 ? $"Process folder ({stale})" : "Up to date";
            RunBtn.IsEnabled = stale > 0;
        }
        bool can = AnyIndexed && !_running;
        SearchBox.IsEnabled = SearchBtn.IsEnabled = ImageBtn.IsEnabled = can;
        SimilarBtn.IsEnabled = can && SegList.SelectedIndex >= 0;
        ClearBtn.IsEnabled = can && _query != null;
    }

    void RefreshVideoList()
    {
        if (_lib == null) return;
        _suppress = true;
        if (_videoVms.Count != _videos.Count)
        {
            _videoVms = _videos.Select(v => new VideoVm(v)).ToList();
            VideoList.ItemsSource = _videoVms;
        }
        foreach (var vm in _videoVms) vm.Notify(); // in place: keeps scroll position and selection
        VideoList.SelectedIndex = _cur == null ? -1 : _videos.IndexOf(_cur);
        _suppress = false;
        UpdateTotals();
    }

    void UpdateTotals()
    {
        if (_lib == null) { TotalText.Text = ""; return; }
        double total = _videos.Where(v => v.Duration > 0).Sum(v => v.Duration);
        int unknown = _videos.Count(v => v.Duration <= 0);
        long size = _videos.Sum(v => v.Size);
        var idx = _videos.Where(v => v.Indexed).ToList();
        TotalText.Text = $"{_videos.Count} videos · {Fmts.Duration(total)}" +
                         (unknown > 0 ? $" (+{unknown} probing…)" : "") + $" · {Fmts.Size(size)}\n" +
                         $"Indexed: {idx.Count} videos · {Fmts.Duration(idx.Sum(v => v.Duration))}";
    }

    /// <summary>Probes duration/resolution of videos that lack it, in the background, updating the list as results arrive.</summary>
    void StartProbing(Library lib)
    {
        _probeCts?.Cancel();
        var cts = _probeCts = new CancellationTokenSource();
        _ = Task.Run(() =>
        {
            try
            {
                lib.ProbeMissing(v => Dispatcher.BeginInvoke(() =>
                {
                    if (_lib != lib) return;
                    _videoVms.FirstOrDefault(x => x.Name == v.Rel)?.Notify();
                    UpdateTotals();
                }), cts.Token);
            }
            catch (OperationCanceledException) { }
            catch (ObjectDisposedException) { }
            catch { /* probing is best-effort */ }
        });
    }

    void VideoList_SelectionChanged(object s, SelectionChangedEventArgs e)
    {
        if (_suppress || VideoList.SelectedIndex < 0) return;
        _cur = _videos[VideoList.SelectedIndex];
        if (_query == null) BuildView();   // browsing: list shows this video's segments
        else { _activeSeg = -1; DrawTimeline(); } // searching: list stays, timeline switches
    }

    // ---- text search ----------------------------------------------------
    void SearchBox_KeyDown(object s, KeyEventArgs e)
    {
        if (e.Key == Key.Enter) Search_Click(s, e);
    }

    async void Search_Click(object s, RoutedEventArgs e) => await RunSearchAsync();

    /// <summary>Runs the query built from the text box and/or the query image (both set = their embeddings are summed).</summary>
    async Task RunSearchAsync()
    {
        var text = SearchBox.Text.Trim();
        if (_running || !AnyIndexed) return;
        if (text.Length == 0 && _imageEmb == null) { Clear_Click(this, new RoutedEventArgs()); return; }

        SetSearchEnabled(false);
        try
        {
            StatusText.Text = "Searching… (models load on first use)";
            var emb = await GetEmbedderAsync();
            var img = _imageEmb;
            _query = await Task.Run(() => Combine(text.Length > 0 ? emb.EmbedText(text) : null, img));
            _queryText = string.Join(" + ", new[]
            {
                text.Length > 0 ? $"“{text}”" : null,
                img != null ? $"image {_imageName}" : null,
            }.Where(x => x != null));
            BuildView();
            if (_view.Count > 0) SegList.SelectedIndex = 0; // jump to and play the best hit
        }
        catch (MissingPrerequisiteException ex) { HandleMissing(ex); }
        catch (Exception ex) { StatusText.Text = "Search failed: " + ex.Message; }
        finally { UpdateButtons(); }
    }

    /// <summary>Sum of the available L2-normalized embeddings, re-normalized.</summary>
    static float[] Combine(float[]? a, float[]? b)
    {
        if (a == null) return b!;
        if (b == null) return a;
        var o = new float[a.Length];
        double ss = 0;
        for (int i = 0; i < o.Length; i++) { o[i] = a[i] + b[i]; ss += (double)o[i] * o[i]; }
        float inv = (float)(1 / Math.Sqrt(ss));
        for (int i = 0; i < o.Length; i++) o[i] *= inv;
        return o;
    }

    void SetSearchEnabled(bool on) => SearchBox.IsEnabled = SearchBtn.IsEnabled = ImageBtn.IsEnabled = SimilarBtn.IsEnabled = on;

    // ---- image search ---------------------------------------------------
    async void Image_Click(object s, RoutedEventArgs e)
    {
        var dlg = new OpenFileDialog { Filter = "Images|*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.gif|All files|*.*" };
        if (dlg.ShowDialog() == true) await SetQueryImageAsync(dlg.FileName);
    }

    async Task SetQueryImageAsync(string path)
    {
        if (_running) return;
        if (!AnyIndexed) { StatusText.Text = "Index some videos first, then search by image."; return; }
        SetSearchEnabled(false);
        try
        {
            StatusText.Text = "Embedding the query image…";
            var emb = await GetEmbedderAsync();
            _imageEmb = await Task.Run(() => emb.EmbedImageFile(path));
            _imageName = System.IO.Path.GetFileName(path);

            var bi = new BitmapImage();
            bi.BeginInit();
            bi.CacheOption = BitmapCacheOption.OnLoad;
            bi.DecodePixelHeight = 60;
            bi.UriSource = new Uri(path);
            bi.EndInit(); bi.Freeze();
            QueryThumb.Source = bi;
            QueryThumb.Visibility = Visibility.Visible;
        }
        catch (Exception ex)
        {
            _imageEmb = null; QueryThumb.Visibility = Visibility.Collapsed;
            StatusText.Text = ex is ModelsUnavailableException ? ex.Message : "Couldn't use that image: " + ex.Message;
            if (ex is MissingPrerequisiteException mp) HandleMissing(mp);
            UpdateButtons();
            return;
        }
        await RunSearchAsync();
    }

    async void QueryThumb_Click(object s, MouseButtonEventArgs e)
    {
        _imageEmb = null; QueryThumb.Visibility = Visibility.Collapsed;
        await RunSearchAsync(); // re-runs with the text only, or clears if there is none
    }

    /// <summary>"More like this": the selected segment's own centroid becomes the query.</summary>
    void Similar_Click(object s, RoutedEventArgs e)
    {
        int sel = SegList.SelectedIndex;
        if (_running || sel < 0 || sel >= _view.Count) return;
        var it = _view[sel];
        _imageEmb = null; QueryThumb.Visibility = Visibility.Collapsed; SearchBox.Text = "";
        _query = (float[])it.V.Segments[it.Seg].Centroid.Clone();
        _queryText = $"similar to {it.V.Rel} #{it.Seg + 1}";
        BuildView();
        UpdateButtons();
    }

    void Clear_Click(object s, RoutedEventArgs e)
    {
        _query = null; _queryText = ""; _imageEmb = null;
        QueryThumb.Visibility = Visibility.Collapsed;
        SearchBox.Text = "";
        if (AnyIndexed) BuildView();
        UpdateButtons();
    }

    // ---- segmentation + list view --------------------------------------
    void Params_Changed(object? s, RoutedEventArgs e)
    {
        UpdateLabels();
        if (IsLoaded && !_running && AnyIndexed) Resegment();
    }

    void UpdateLabels()
    {
        if (ThrLabel == null || MaxLabel == null || NoLimitBox == null) return; // events fire during InitializeComponent
        ThrLabel.Text = $"Similarity threshold: {ThrSlider.Value:F2}";
        // Hidden (not Collapsed) so the column doesn't change height when toggling.
        MaxSlider.Visibility = NoLimitBox.IsChecked == true ? Visibility.Hidden : Visibility.Visible;
        MaxLabel.Text = NoLimitBox.IsChecked == true ? "Max segment length: unlimited" : $"Max segment length: {MaxSlider.Value:F0} s";
    }

    /// <summary>Re-segments every indexed video with the current parameters, then rebuilds the list.</summary>
    void Resegment()
    {
        Library.Resegment(_videos, ThrSlider.Value, MaxSeconds, BlipBox.IsChecked == true);
        BuildView();
    }

    void BuildView()
    {
        StopPlayback();
        _activeSeg = -1;
        _view = [];
        int total = 0;

        if (_query != null)
        {
            var (top, tot) = Library.Search(_videos, _query, TopN);
            total = tot;
            _view = top.Select((h, r) => new ViewItem(h.Video, h.SegmentIndex, h.Score, h.BestFrame, r)).ToList();
            if (_view.Count > 0) _cur = _view[0].V;
        }
        else if (_cur is { Indexed: true })
        {
            _view = _cur.Segments.Select((_, i) => new ViewItem(_cur, i, 0, -1, i)).ToList();
        }

        SegList.ItemsSource = _view.Select(MakeVm).ToList();
        SyncVideoSelection();
        DrawTimeline();

        if (_query != null)
            StatusText.Text = $"Query {_queryText}: top {_view.Count} of {total} segments in {_videos.Count(v => v.Indexed)} video(s).";
        else if (_cur is { Indexed: true })
            StatusText.Text = $"{_cur.Rel}: {_cur.Frames.Count} frames → {_cur.Segments.Count} segments" +
                              (_cur.Segments.Count > 0 ? $" (avg {_cur.Segments.Average(x => x.Duration):F1} s)" : "");
    }

    void SyncVideoSelection()
    {
        if (_lib == null) return;
        _suppress = true;
        VideoList.SelectedIndex = _cur == null ? -1 : _videos.IndexOf(_cur);
        _suppress = false;
    }

    SegmentVm MakeVm(ViewItem it)
    {
        var v = it.V; var g = v.Segments[it.Seg];
        string title = $"#{it.Seg + 1}   {Fmt(g.Start)} – {Fmt(g.End)}";
        string info = $"{g.Duration:F1} s · {g.FrameCount} frames";
        if (_query != null)
        {
            title = $"[{it.Rank + 1}]  {v.Rel}   " + title;
            info += $" · score {it.Score:F3} · best match at {Fmt(v.Frames[it.Best].T)}";
        }
        return new SegmentVm(title, info, () => ThumbStrip(v, g));
    }

    List<BitmapSource> ThumbStrip(LibraryVideo v, Segment g)
    {
        int n = Math.Min(6, g.FrameCount);
        var list = new List<BitmapSource>();
        for (int k = 0; k < n; k++)
        {
            int idx = g.FirstFrame + (n == 1 ? 0 : (int)Math.Round((double)k * (g.FrameCount - 1) / (n - 1)));
            var b = ThumbFor(v, idx);
            if (b != null) list.Add(b);
        }
        return list;
    }

    BitmapSource? ThumbFor(LibraryVideo v, int idx)
    {
        var raw = v.Frames[idx].Thumb;
        if (raw.Length > 0)
        {
            int n = FrameSampler.ThumbSize;
            var bmp = BitmapSource.Create(n, n, 96, 96, PixelFormats.Rgb24, null, raw, n * 3);
            bmp.Freeze();
            return bmp;
        }
        var jpg = _lib?.GetThumb(v, idx);
        if (jpg == null) return null;
        var bi = new BitmapImage();
        bi.BeginInit();
        bi.CacheOption = BitmapCacheOption.OnLoad;
        bi.StreamSource = new MemoryStream(jpg);
        bi.EndInit();
        bi.Freeze();
        return bi;
    }

    static string Fmt(double sec)
    {
        var t = TimeSpan.FromSeconds(sec);
        return t.TotalHours >= 1 ? t.ToString(@"h\:mm\:ss\.f", CultureInfo.InvariantCulture) : t.ToString(@"mm\:ss\.f", CultureInfo.InvariantCulture);
    }

    // ---- timeline (shows _cur) -----------------------------------------
    void Timeline_SizeChanged(object s, SizeChangedEventArgs e) => DrawTimeline();

    void DrawTimeline()
    {
        Timeline.Children.Clear(); _rects.Clear();
        double w = Timeline.ActualWidth, h = Timeline.ActualHeight;
        if (_cur is not { Indexed: true } cur || cur.Segments.Count == 0 || cur.Duration <= 0 || w <= 0) return;

        for (int i = 0; i < cur.Segments.Count; i++)
        {
            var g = cur.Segments[i];
            double x0 = g.Start / cur.Duration * w, x1 = g.End / cur.Duration * w;
            var r = new Rectangle
            {
                Width = Math.Max(1, x1 - x0),
                Height = h,
                Fill = new SolidColorBrush(Palette[i % Palette.Length]),
                Stroke = new SolidColorBrush(Color.FromRgb(0x22, 0x22, 0x22)),
                StrokeThickness = 1,
                Tag = i,
                ToolTip = $"#{i + 1}  {Fmt(g.Start)} – {Fmt(g.End)}  ({g.Duration:F1} s)",
                Cursor = Cursors.Hand,
            };
            r.MouseLeftButtonDown += (_, _) => TimelineClick((int)r.Tag);
            Canvas.SetLeft(r, x0);
            Timeline.Children.Add(r);
            _rects.Add(r);
        }
        HighlightSelected();
    }

    void TimelineClick(int segIdx)
    {
        if (_cur == null) return;
        int pos = _view.FindIndex(x => x.V == _cur && x.Seg == segIdx);
        if (pos >= 0) { SegList.SelectedIndex = pos; SegList.ScrollIntoView(SegList.SelectedItem); }
        else { _activeSeg = segIdx; HighlightSelected(); PlaySegment(_cur, _cur.Segments[segIdx]); } // not in the result list: just play it
    }

    void HighlightSelected()
    {
        // In search mode only the segments of the current video that are among the results stay bright.
        HashSet<int>? bright = _query == null ? null
            : _view.Where(x => x.V == _cur && x.Rank < TopK).Select(x => x.Seg).ToHashSet();
        for (int i = 0; i < _rects.Count; i++)
        {
            bool on = i == _activeSeg;
            bool hit = bright == null || bright.Contains(i);
            _rects[i].Stroke = on ? Brushes.White : new SolidColorBrush(Color.FromRgb(0x22, 0x22, 0x22));
            _rects[i].StrokeThickness = on ? 3 : 1;
            _rects[i].Opacity = on ? 1.0 : hit ? (_activeSeg < 0 ? 1.0 : 0.7) : 0.2;
            Panel.SetZIndex(_rects[i], on ? 1 : 0);
        }
    }

    // ---- playback -------------------------------------------------------
    void SegList_SelectionChanged(object s, SelectionChangedEventArgs e)
    {
        int sel = SegList.SelectedIndex;
        SimilarBtn.IsEnabled = sel >= 0 && !_running && AnyIndexed;
        if (sel < 0 || sel >= _view.Count) return;
        var it = _view[sel];
        bool videoChanged = _cur != it.V;
        _cur = it.V;
        _activeSeg = it.Seg;
        if (videoChanged) { SyncVideoSelection(); DrawTimeline(); } else HighlightSelected();
        PlaySegment(it.V, it.V.Segments[it.Seg]);
    }

    const string IconPause = "", IconPlay = ""; // Segoe MDL2 Assets
    LibraryVideo? _playingVideo;
    bool _scrubbing;

    void PlaySegment(LibraryVideo v, Segment seg)
    {
        try { Prerequisites.RequireFfmpeg(); }
        catch (MissingPrerequisiteException ex) { HandleMissing(ex); return; }

        // The decode size depends on the source size; probe it now if the background probing hasn't reached this file yet.
        if (v.Width <= 0) { var info = VideoIndexer.Probe(v.FullPath); v.Width = info.Width; v.Height = info.Height; }

        _playing = seg; _playingVideo = v;
        SetPauseIcon(false); PauseBtn.IsEnabled = true;
        SeekBar.Maximum = Math.Max(0.5, seg.End - seg.Start); SeekBar.Value = 0; SeekBar.IsEnabled = true;
        PlayerLabel.Text = $"{v.Rel}: {Fmt(seg.Start)} - {Fmt(seg.End)} (loops)";
        _player.Muted = MuteBox.IsChecked == true;
        _player.Play(v.FullPath, seg.Start, seg.End, v.Width, v.Height);
        _timer.Start();
    }

    void SetPauseIcon(bool paused)
    {
        PauseBtn.Content = paused ? IconPlay : IconPause;
        PauseBtn.ToolTip = paused ? "Play" : "Pause";
    }

    void Player_Failed(string message)
    {
        _timer.Stop();
        PlayerLabel.Text = message;
        PauseBtn.IsEnabled = false; SeekBar.IsEnabled = false;
    }

    void ShowPosition(double pos) =>
        PlayerLabel.Text = $"{Fmt(Math.Clamp(pos, _playing!.Start, _playing.End))}  in  {Fmt(_playing.Start)} - {Fmt(_playing.End)} (loops)";

    void Timer_Tick(object? s, EventArgs e)
    {
        if (_playing == null || _player.Paused || _scrubbing) return;
        double pos = _player.Position;
        ShowPosition(pos);
        SeekBar.Value = Math.Clamp(pos - _playing.Start, 0, SeekBar.Maximum);
    }

    void SeekBar_MouseDown(object s, MouseButtonEventArgs e)
    {
        if (_playing == null) return;
        _scrubbing = true;
    }

    void SeekBar_ValueChanged(object s, RoutedPropertyChangedEventArgs<double> e)
    {
        if (_scrubbing && _playing != null) ShowPosition(_playing.Start + SeekBar.Value);
    }

    void SeekBar_MouseUp(object s, MouseButtonEventArgs e)
    {
        if (!_scrubbing) return;
        _scrubbing = false;
        if (_playing == null || _playingVideo == null) return;
        var v = _playingVideo;
        _player.Play(v.FullPath, _playing.Start, _playing.End, v.Width, v.Height, SeekBar.Value, keepPaused: true);
        ShowPosition(_playing.Start + SeekBar.Value);
    }

    void Pause_Click(object s, RoutedEventArgs e)
    {
        if (_playing == null) return;
        _player.Paused = !_player.Paused;
        SetPauseIcon(_player.Paused);
    }

    void Mute_Changed(object s, RoutedEventArgs e)
    {
        bool muted = MuteBox.IsChecked == true;
        _player.Muted = muted;
        MuteBox.Content = muted ? "\uE74F" : "\uE767"; // Segoe MDL2 Assets: mute / volume
        MuteBox.ToolTip = muted ? "Unmute" : "Mute";
    }

    void StopPlayback()
    {
        _timer.Stop();
        _player.Stop();
        PlayerImage.Source = null;
        _playing = null; _playingVideo = null; _scrubbing = false;
        SetPauseIcon(false); PauseBtn.IsEnabled = false;
        SeekBar.Value = 0; SeekBar.IsEnabled = false;
        PlayerLabel.Text = "Select a segment to play it (loops).";
    }

    protected override void OnClosed(EventArgs e)
    {
        _cts?.Cancel();
        _probeCts?.Cancel();
        _timer.Stop();
        _player.Dispose();
        _lib?.Dispose();
        _emb?.Dispose();
        base.OnClosed(e);
    }
}
