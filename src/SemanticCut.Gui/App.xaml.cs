using System.Windows;

namespace SemanticCut.Gui;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        // Always format numbers with '.' regardless of the system locale (UI thread + all thread-pool threads).
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        System.Globalization.CultureInfo.DefaultThreadCurrentCulture = inv;
        System.Globalization.CultureInfo.DefaultThreadCurrentUICulture = inv;
        Thread.CurrentThread.CurrentCulture = inv;
        Thread.CurrentThread.CurrentUICulture = inv;

        // Show unhandled UI exceptions instead of silently dying.
        DispatcherUnhandledException += (_, ev) =>
        {
            MessageBox.Show(ev.Exception.ToString(), "Semantic Cut - unhandled error");
            ev.Handled = true;
        };
        base.OnStartup(e);
    }
}
