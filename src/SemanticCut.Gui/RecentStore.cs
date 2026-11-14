using System.IO;
using System.Text.Json;

namespace SemanticCut.Gui;

/// <summary>
/// The most recently opened folders and video files, newest first. Stored per user in
/// %LOCALAPPDATA%\SemanticCut\recent.json (never inside the repo or next to the exe).
/// </summary>
static class RecentStore
{
    const int Max = 12;

    static string FilePath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SemanticCut", "recent.json");

    public static List<string> Load()
    {
        try
        {
            if (File.Exists(FilePath))
                return JsonSerializer.Deserialize<List<string>>(File.ReadAllText(FilePath)) ?? [];
        }
        catch { /* unreadable list: start empty */ }
        return [];
    }

    /// <summary>Moves <paramref name="path"/> to the top of the list (adding it if new).</summary>
    public static void Add(string path)
    {
        var list = Load();
        list.RemoveAll(p => string.Equals(p, path, StringComparison.OrdinalIgnoreCase));
        list.Insert(0, path);
        Save(list.Take(Max).ToList());
    }

    public static void Remove(string path)
    {
        var list = Load();
        list.RemoveAll(p => string.Equals(p, path, StringComparison.OrdinalIgnoreCase));
        Save(list);
    }

    public static void Clear() => Save([]);

    static void Save(List<string> list)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
            File.WriteAllText(FilePath, JsonSerializer.Serialize(list));
        }
        catch { /* the list is a convenience; never fail an open because of it */ }
    }
}
