using System.Runtime.InteropServices;
using Microsoft.Data.Sqlite;
using SkiaSharp;

namespace SemanticCut.Core;

public sealed record VideoRow(int Id, string Path, long Size, long Mtime, double Duration, double Every, int Width, int Height);

public static class FolderScanner
{
    static readonly HashSet<string> Ext = new(StringComparer.OrdinalIgnoreCase)
    {
        ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".m2ts", ".mxf", ".mpg", ".mpeg", ".wmv", ".ts",
    };

    /// <summary>Recursively finds video files; skips hidden folders. Returned paths are relative with '/' separators.</summary>
    public static List<(string rel, FileInfo file)> Scan(string folder)
    {
        var opts = new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = true, AttributesToSkip = FileAttributes.Hidden | FileAttributes.System };
        return new DirectoryInfo(folder).EnumerateFiles("*", opts)
            .Where(f => Ext.Contains(f.Extension))
            .Select(f => (Path.GetRelativePath(folder, f.FullName).Replace('\\', '/'), f))
            .OrderBy(x => x.Item1, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }
}

/// <summary>
/// Per-folder SQLite index (semantic_cut.db). Stores, per video, every sampled frame as (t, float16 embedding, JPEG thumbnail).
/// Segments are NOT stored: they are derived from the frames with the current parameters.
/// </summary>
public sealed class LibraryDb : IDisposable
{
    public const string FileName = "semantic_cut.db";

    readonly SqliteConnection _c;
    readonly object _lock = new();
    public string Folder { get; }

    public LibraryDb(string folder)
    {
        Folder = folder;
        _c = new SqliteConnection($"Data Source={Path.Combine(folder, FileName)}");
        _c.Open();
        Exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
        Exec("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS videos(
              id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, size INTEGER NOT NULL, mtime INTEGER NOT NULL,
              duration REAL NOT NULL, every REAL NOT NULL, indexed_at TEXT NOT NULL,
              width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS frames(
              video_id INTEGER NOT NULL, idx INTEGER NOT NULL, t REAL NOT NULL, emb BLOB NOT NULL, thumb BLOB,
              PRIMARY KEY(video_id, idx));
            """);

        // Migration: databases created before resolution was stored.
        bool hasWidth = false;
        using (var pragma = _c.CreateCommand())
        {
            pragma.CommandText = "PRAGMA table_info(videos)";
            using var rd = pragma.ExecuteReader();
            while (rd.Read()) if (rd.GetString(1) == "width") hasWidth = true;
        }
        if (!hasWidth)
            Exec("ALTER TABLE videos ADD COLUMN width INTEGER NOT NULL DEFAULT 0; ALTER TABLE videos ADD COLUMN height INTEGER NOT NULL DEFAULT 0;");

        using var cmd = _c.CreateCommand();
        cmd.CommandText = "SELECT value FROM meta WHERE key='model_id'";
        var existing = cmd.ExecuteScalar() as string;
        if (existing == null)
        {
            cmd.CommandText = "INSERT INTO meta(key,value) VALUES('model_id',$m),('dim',$d)";
            cmd.Parameters.AddWithValue("$m", SigLipEmbedder.ModelId);
            cmd.Parameters.AddWithValue("$d", SigLipEmbedder.Dim.ToString());
            cmd.ExecuteNonQuery();
        }
        else if (existing != SigLipEmbedder.ModelId)
            throw new InvalidOperationException($"This folder's index was built with '{existing}', but the app uses '{SigLipEmbedder.ModelId}'. Delete {FileName} to re-index.");
    }

    void Exec(string sql)
    {
        using var cmd = _c.CreateCommand();
        cmd.CommandText = sql;
        cmd.ExecuteNonQuery();
    }

    public Dictionary<string, VideoRow> GetVideos()
    {
        lock (_lock)
        {
            using var cmd = _c.CreateCommand();
            cmd.CommandText = "SELECT id,path,size,mtime,duration,every,width,height FROM videos";
            using var r = cmd.ExecuteReader();
            var d = new Dictionary<string, VideoRow>();
            while (r.Read())
                d[r.GetString(1)] = new VideoRow(r.GetInt32(0), r.GetString(1), r.GetInt64(2), r.GetInt64(3), r.GetDouble(4),
                    r.GetDouble(5), r.GetInt32(6), r.GetInt32(7));
            return d;
        }
    }

    public void UpdateVideoMeta(int id, int width, int height)
    {
        lock (_lock)
        {
            using var cmd = _c.CreateCommand();
            cmd.CommandText = "UPDATE videos SET width=$w, height=$h WHERE id=$id";
            cmd.Parameters.AddWithValue("$w", width);
            cmd.Parameters.AddWithValue("$h", height);
            cmd.Parameters.AddWithValue("$id", id);
            cmd.ExecuteNonQuery();
        }
    }

    /// <summary>Replaces any previous index of this path, in a single transaction. JPEG encoding happens outside the DB lock.</summary>
    public int SaveVideo(string relPath, long size, long mtime, double duration, double every, int width, int height,
        IReadOnlyList<FrameRecord> frames)
    {
        var rows = new (double t, byte[] emb, byte[]? thumb)[frames.Count];
        Parallel.For(0, frames.Count, i =>
        {
            var f = frames[i];
            rows[i] = (f.T, EmbToBlob(f.Emb), f.Thumb.Length > 0 ? ToJpeg(f.Thumb) : null);
        });

        lock (_lock)
        {
            using var tx = _c.BeginTransaction();
            using (var del = _c.CreateCommand())
            {
                del.Transaction = tx;
                del.CommandText = "DELETE FROM frames WHERE video_id IN (SELECT id FROM videos WHERE path=$p); DELETE FROM videos WHERE path=$p;";
                del.Parameters.AddWithValue("$p", relPath);
                del.ExecuteNonQuery();
            }
            int id;
            using (var ins = _c.CreateCommand())
            {
                ins.Transaction = tx;
                ins.CommandText = "INSERT INTO videos(path,size,mtime,duration,every,indexed_at,width,height) VALUES($p,$s,$m,$d,$e,$a,$w,$h); SELECT last_insert_rowid();";
                ins.Parameters.AddWithValue("$w", width);
                ins.Parameters.AddWithValue("$h", height);
                ins.Parameters.AddWithValue("$p", relPath);
                ins.Parameters.AddWithValue("$s", size);
                ins.Parameters.AddWithValue("$m", mtime);
                ins.Parameters.AddWithValue("$d", duration);
                ins.Parameters.AddWithValue("$e", every);
                ins.Parameters.AddWithValue("$a", DateTime.UtcNow.ToString("O"));
                id = Convert.ToInt32(ins.ExecuteScalar());
            }
            using (var fr = _c.CreateCommand())
            {
                fr.Transaction = tx;
                fr.CommandText = "INSERT INTO frames(video_id,idx,t,emb,thumb) VALUES($v,$i,$t,$e,$th)";
                var pv = fr.Parameters.Add("$v", SqliteType.Integer); pv.Value = id;
                var pi = fr.Parameters.Add("$i", SqliteType.Integer);
                var pt = fr.Parameters.Add("$t", SqliteType.Real);
                var pe = fr.Parameters.Add("$e", SqliteType.Blob);
                var pth = fr.Parameters.Add("$th", SqliteType.Blob);
                for (int i = 0; i < rows.Length; i++)
                {
                    pi.Value = i; pt.Value = rows[i].t; pe.Value = rows[i].emb;
                    pth.Value = (object?)rows[i].thumb ?? DBNull.Value;
                    fr.ExecuteNonQuery();
                }
            }
            tx.Commit();
            return id;
        }
    }

    /// <summary>Loads timestamps + embeddings (no thumbnails; Thumb is empty and fetched lazily via GetThumb).</summary>
    public List<FrameRecord> LoadFrames(int videoId)
    {
        lock (_lock)
        {
            using var cmd = _c.CreateCommand();
            cmd.CommandText = "SELECT t,emb FROM frames WHERE video_id=$v ORDER BY idx";
            cmd.Parameters.AddWithValue("$v", videoId);
            using var r = cmd.ExecuteReader();
            var list = new List<FrameRecord>();
            while (r.Read())
                list.Add(new FrameRecord(r.GetDouble(0), BlobToEmb(r.GetFieldValue<byte[]>(1)), []));
            return list;
        }
    }

    public byte[]? GetThumb(int videoId, int idx)
    {
        lock (_lock)
        {
            using var cmd = _c.CreateCommand();
            cmd.CommandText = "SELECT thumb FROM frames WHERE video_id=$v AND idx=$i";
            cmd.Parameters.AddWithValue("$v", videoId);
            cmd.Parameters.AddWithValue("$i", idx);
            return cmd.ExecuteScalar() as byte[];
        }
    }

    static byte[] EmbToBlob(float[] e)
    {
        var h = new Half[e.Length];
        for (int i = 0; i < e.Length; i++) h[i] = (Half)e[i];
        return MemoryMarshal.AsBytes(h.AsSpan()).ToArray();
    }

    static float[] BlobToEmb(byte[] b)
    {
        var h = MemoryMarshal.Cast<byte, Half>(b);
        var e = new float[h.Length];
        for (int i = 0; i < e.Length; i++) e[i] = (float)h[i];
        return e;
    }

    static byte[] ToJpeg(byte[] rgb)
    {
        int n = FrameSampler.ThumbSize;
        var rgba = new byte[n * n * 4];
        for (int i = 0; i < n * n; i++)
        {
            rgba[i * 4] = rgb[i * 3]; rgba[i * 4 + 1] = rgb[i * 3 + 1]; rgba[i * 4 + 2] = rgb[i * 3 + 2]; rgba[i * 4 + 3] = 255;
        }
        using var bmp = new SKBitmap(new SKImageInfo(n, n, SKColorType.Rgba8888, SKAlphaType.Opaque));
        Marshal.Copy(rgba, 0, bmp.GetPixels(), rgba.Length);
        using var img = SKImage.FromBitmap(bmp);
        using var data = img.Encode(SKEncodedImageFormat.Jpeg, 80);
        return data.ToArray();
    }

    public void Dispose()
    {
        lock (_lock) { _c.Dispose(); }
    }
}
