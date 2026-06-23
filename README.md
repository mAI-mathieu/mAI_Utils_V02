# mAI ComfyUI Node Template

Base folder for creating a new ComfyUI custom node pack.

Copy this folder into:

```text
C:\AI\ComfyUIOpti\ComfyUICuda13\ComfyUI\custom_nodes
```

Then rename it, for example:

```text
mAI_ImageBrowser
mAI_BlurAwareMaskTools
mAI_FrameCanvasRecut
```

## Included

```text
__init__.py                    ComfyUI node registration
nodes/example_text_node.py      Minimal example node
utils/                          Shared helper code folder
web/js/                         Optional ComfyUI frontend extension folder
examples/                       Example workflows or screenshots
tests/                          Small pure Python tests
docs/                           Notes and Codex prompts
AGENTS.md                       Instructions Codex should follow
HOWTO_GIT_CODEX.md              Setup guide for Git and Codex
SETUP_GIT_FOR_CODEX.bat         Easiest Windows setup script for Git + GitHub + Codex
scripts/init_git.ps1            Optional PowerShell helper script for Git setup
```

## Minimal node behavior

The example node takes a string and returns a cleaned string.
It is intentionally simple so you can replace it with the real node logic.

After copying this folder into `custom_nodes`, restart ComfyUI and look for:

```text
mAI / Template / Example Text Node
```

## mAI Composite Layer

Location:

```text
mAI / Image
```

Purpose:
Composites one image layer over one base image.
Use multiple copies of the node chained together to create multi-layer compositions.

Inputs:

* `base_image`
* optional `base_mask`
* `layer_image`
* optional `layer_mask`

Widgets:

* `x`
* `y`
* `scale`
* `opacity`
* `anchor`
* `fit_mode`

Outputs:

* `image`
* `mask`

Transparency:

* Uses `layer_mask` if connected.
* Otherwise uses embedded alpha if the layer image has 4 channels.
* Otherwise treats the layer as opaque.

Mask:

* The mask output is the placed foreground alpha.
* If `base_mask` is connected, the output mask combines `base_mask` and the current layer alpha.
* This makes it possible to chain masks across multiple Composite Layer nodes.

Why this replaced Image Layer Stack:

The old 8-layer fixed stack created a huge ComfyUI node.
The new chainable node is smaller, easier to control, and supports unlimited layers by duplication.

Notes:

* Output is RGB.
* Transparent PNG style layers are supported when ComfyUI passes alpha channels.
* Negative `x` and `y` are supported.
* Layers can be partially outside the background.
* The old `mAI Image Layer Stack` implementation remains in the codebase but is not registered by default.

Test command:

```powershell
python -m pytest
```

## mAI Save Text File

Location:

```text
mAI / IO
```

Purpose:
Saves a text string to a file using UTF-8 encoding.

Inputs:

* `text`
* `file_name`
* `folder`
* `overwrite`

Outputs:

* `file_path`
* `saved`

Default behavior:

* Saves into the current working directory when `folder` is empty.
* Creates `folder` if it does not exist.
* Uses the exact provided `file_name`.
* Does not add timestamps or change the file name.
* Does not auto-add `.txt`; include `.txt` in `file_name` when wanted.
* Raises `FileExistsError` when `overwrite` is false and the target file already exists.

Known limitations:

* `file_name` must be a simple file name, not a nested or absolute path.
* Put folder paths in `folder`, not in `file_name`.

Test command:

```powershell
python -m pytest
```

## mAI Random Line

Location:

```text
mAI / Utils
```

Purpose:
Returns one random non-empty line from multiline text.

Inputs:

* `text`

Outputs:

* `line`

Default behavior:

* Uses Windows and Unix line endings.
* Ignores empty and whitespace-only lines.
* Preserves the selected line text except for the removed line break.
* Returns an empty string when there are no non-empty lines.
* Produces a fresh random choice on each queued execution.

Known limitations:

* There is no visible seed input, so results are intentionally not reproducible.

Test command:

```powershell
python -m pytest
```

## Development rule

Keep each custom node pack as its own Git repo.
Do not make your full ComfyUI install the repo.


## Fast Windows setup

After copying and renaming this folder, double-click:

```text
SETUP_GIT_FOR_CODEX.bat
```

It will initialize Git, create the `main` branch, commit the template, ask for your GitHub remote URL, and push if possible.

It cannot create the GitHub repository for you unless you use extra tools like GitHub CLI, so create an empty repo on GitHub first.


## One-click GitHub setup

The BAT file can create the GitHub repo automatically if GitHub CLI is installed and authenticated:

```powershell
winget install --id GitHub.cli
gh auth login
```

Then double-click `SETUP_GIT_FOR_CODEX.bat`. It can create the remote repo, add it as `origin`, and push `main`.
