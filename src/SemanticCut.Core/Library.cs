namespace SemanticCut.Core;

/// <summary>A video known to the app: either backed by a folder DB, or an ad-hoc single file (no DB).</summary>
public sealed class LibraryVideo
{
    public required string Rel { get; init; }        // name shown in the UI (relative path for folders)
    public required string FullPath { get; init; }
    public long Size { get; set; }
    public long Mtime { get; set; }
    public int DbId { get; set; }
    public double Duration { get; set; }             // 0 = not probed yet
    public int Width { get; set; }                   // 0 = not probed yet
    public int Height { get; set; }
    public double Every { get; set; } = 2;
    public List<FrameRecord> Frames { get; set; } = [];
    public List<Segment> Segments { get; set; } = [];
    public bool Stale { get; set; } = true;          // needs (re)indexing
    public bool Indexed => Frames.Count > 0;
}

public sealed record SearchHit(LibraryVideo Video, int SegmentIndex, float Score, int BestFrame)
{
    public Segment Segment => Video.Segments[SegmentIndex];
    public double BestTime => Video.Frames[BestFrame].T;
}

public sealed class Library : IDisposable
{
    public LibraryDb Db { get; }
    public List<LibraryVideo> Videos { get; }
    public string Folder => Db.Folder;

    Library(LibraryDb db, List<LibraryVideo> videos) { Db = db; Videos = videos; }

    /// <summary>Opens (creating if needed) the folder's DB, scans for videos and loads embeddings of those already indexed and unchanged.</summary>
    public static Library Open(string folder)
    {
        var db = new LibraryDb(folder);
        var rows = db.GetVideos();
        var list = new List<LibraryVideo>();
        foreach (var (rel, fi) in FolderScanner.Scan(folder))
        {
            var v = new LibraryVideo { Rel = rel, FullPath = fi.FullName, Size = fi.Length, Mtime = fi.LastWriteTimeUtc.Ticks };
            if (rows.TryGetValue(rel, out var r) && r.Size == v.Size && r.Mtime == v.Mtime)
            {
                v.DbId = r.Id; v.Duration = r.Duration; v.Every = r.Every; v.Width = r.Width; v.Height = r.Height;
                v.Frames = db.LoadFrames(r.Id);
                v.Stale = v.Frames.Count == 0;
            }
            list.Add(v);
        }
        return new Library(db, list);
    }

    /// <summary>Embeds the video and replaces its DB rows. Frames in memory keep embeddings only (thumbnails live in the DB).</summary>
    public void Index(LibraryVideo v, SigLipEmbedder emb, double every, IProgress<(int done, int total)>? progress, CancellationToken ct)
    {
        Prerequisites.RequireFfmpeg(); // fail with a clear message before probing silently returns nothing
        var info = VideoIndexer.Probe(v.FullPath);
        double duration = info.Duration;
        var frames = VideoIndexer.Run(emb, v.FullPath, every, duration, progress, ct);
        if (duration <= 0 && frames.Count > 0) duration = frames[^1].T + every;
        v.DbId = Db.SaveVideo(v.Rel, v.Size, v.Mtime, duration, every, info.Width, info.Height, frames);
        v.Duration = duration; v.Every = every; v.Width = info.Width; v.Height = info.Height;
        v.Frames = frames.Select(f => f with { Thumb = [] }).ToList();
        v.Segments = [];
        v.Stale = frames.Count == 0;
    }

    /// <summary>Fills in duration/resolution for videos that don't have them yet (not indexed, or indexed before resolution was stored).</summary>
    public void ProbeMissing(Action<LibraryVideo>? onProbed, CancellationToken ct)
    {
        var todo = Videos.Where(v => v.Duration <= 0 || v.Width <= 0).ToList();
        Parallel.ForEach(todo, new ParallelOptions { MaxDegreeOfParallelism = 4, CancellationToken = ct }, v =>
        {
            var info = VideoIndexer.Probe(v.FullPath);
            if (v.Duration <= 0) v.Duration = info.Duration;
            v.Width = info.Width; v.Height = info.Height;
            if (v.DbId > 0 && info.Width > 0) Db.UpdateVideoMeta(v.DbId, info.Width, info.Height);
            onProbed?.Invoke(v);
        });
    }

    public byte[]? GetThumb(LibraryVideo v, int frameIdx) => Db.GetThumb(v.DbId, frameIdx);

    public static void Resegment(IEnumerable<LibraryVideo> videos, double threshold, double maxSeconds, bool ignoreBlips)
    {
        Parallel.ForEach(videos.Where(v => v.Indexed), v =>
            v.Segments = Segmenter.Build(v.Frames, v.Every, threshold, maxSeconds, ignoreBlips, v.Duration));
    }

    /// <summary>Global top-N segments across videos, ranked by best-matching frame. Videos must have Segments built.</summary>
    public static (List<SearchHit> top, int totalSegments) Search(IEnumerable<LibraryVideo> videos, float[] query, int topN)
    {
        var all = new List<SearchHit>();
        foreach (var v in videos.Where(v => v.Indexed))
        {
            var scores = SegmentSearch.Score(v.Frames, v.Segments, query);
            for (int i = 0; i < scores.Length; i++) all.Add(new SearchHit(v, i, scores[i].score, scores[i].bestFrame));
        }
        return (all.OrderByDescending(h => h.Score).Take(topN).ToList(), all.Count);
    }

    public void Dispose() => Db.Dispose();
}
