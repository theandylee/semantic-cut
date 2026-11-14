# semcut

Semantic search over video footage: embeds sampled frames with SigLIP 2, groups similar neighbours into segments,
and lets you search a whole folder by text, by image, or by "more like this" — from a GUI (`semcut-gui.exe`) or a CLI (`semcut.exe`).

## Requirements (Windows x64, NVIDIA GPU)

| Needed | Where to get it | If missing |
|---|---|---|
| **.NET 10 Desktop Runtime** (x64) | <https://dotnet.microsoft.com/download/dotnet/10.0> | The exe itself says so and links to the download. |
| **ffmpeg + ffprobe** on `PATH` (or next to the exe) | <https://www.gyan.dev/ffmpeg/builds/> or `winget install Gyan.FFmpeg` | The app says so, with the link. |
| **NVIDIA driver + CUDA Toolkit 12.x** | <https://developer.nvidia.com/cuda-12-8-0-download-archive> | The app says so, with the link. |
| SigLIP 2 model files + cuDNN (~3 GB) | Downloaded automatically into `models\` next to the exe on first use (asks first in the GUI). | — |

## Build

```
dotnet build -c Release
```

Everything (both exes and all DLLs) goes into `build\`. Close both apps before rebuilding, since they lock the shared DLLs.

## Use

```
semcut-gui.exe [folder]                       # GUI; drop a folder or video on the window
semcut index  <folder>                        # index new/changed videos into <folder>\semantic_cut.db
semcut find   <folder> "text or image.jpg"    # search all indexed videos
semcut download                               # fetch the models + cuDNN explicitly
```

Run `semcut` with no arguments for the full command list.

In the GUI, selected segments play inside the window (ffmpeg decodes video and audio, so any format ffmpeg reads works).

## Layout

- `src/SemanticCut.Core` – embedding (ONNX Runtime), frame sampling, segmentation, SQLite library, model downloader
- `src/SemanticCut.Cli` – `semcut.exe`
- `src/SemanticCut.Gui` – `semcut-gui.exe` (WPF)

NuGet packages are restored by `dotnet build`; model files and cuDNN are downloaded by the program. Neither is stored in this repository.
The code is MIT-licensed (see `LICENSE`). Model weights (SigLIP 2, Apache-2.0) and cuDNN/CUDA have their own licences.

Models are looked up in `--models <path>`, then `%SEMCUT_MODELS%`, then `models\` next to the exe.
