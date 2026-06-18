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
