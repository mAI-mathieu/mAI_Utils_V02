# How to start a new ComfyUI custom node repo with Git and Codex

This guide assumes your ComfyUI path is:

```text
C:\AI\ComfyUIOpti\ComfyUICuda13\ComfyUI
```

Adapt the path if your install is somewhere else.

## 1. Copy and rename the template folder

Copy this folder into:

```text
C:\AI\ComfyUIOpti\ComfyUICuda13\ComfyUI\custom_nodes
```

Rename the copied folder to the real node pack name, for example:

```text
mAI_ImageBrowser
```

Your final path should look like:

```text
C:\AI\ComfyUIOpti\ComfyUICuda13\ComfyUI\custom_nodes\mAI_ImageBrowser
```

## 2. Fast setup with the BAT file

After copying and renaming the template folder, double-click:

```text
SETUP_GIT_FOR_CODEX.bat
```

The BAT file does this for you:

```text
- checks that Git is installed
- adds the folder as a Git safe directory
- initializes Git if needed
- creates or renames the branch to main
- creates the first commit
- can create the GitHub repo automatically with GitHub CLI, gh
- adds the GitHub repo as origin
- pushes main to GitHub
```

## 3. Recommended automatic GitHub setup

Install GitHub CLI once:

```powershell
winget install --id GitHub.cli
```

Then authenticate once:

```powershell
gh auth login
```

After that, the BAT file can create the GitHub repo automatically.

When asked:

```text
Create GitHub repo automatically with gh? [Y/n]
```

Press Enter.

Then choose:

```text
Repo name: press Enter to use the folder name
Visibility: private or public
Owner or org: leave empty for your personal GitHub account
```

The script runs the equivalent of:

```powershell
gh repo create YOUR_REPO --private --source . --remote origin --push
```

That creates the remote repository, adds it as `origin`, and pushes your local commit.

## 4. Manual fallback

If you do not want GitHub CLI, the script can still ask you for a remote URL.

Create an empty GitHub repo manually, then paste a URL like:

```text
https://github.com/YOUR_USERNAME/mAI_ImageBrowser.git
```

or:

```text
git@github.com:YOUR_USERNAME/mAI_ImageBrowser.git
```

## 5. Check that the repo is correct

Open PowerShell in the node folder:

```powershell
cd C:\AI\ComfyUIOpti\ComfyUICuda13\ComfyUI\custom_nodes\mAI_ImageBrowser
```

Run:

```powershell
git rev-parse --show-toplevel
git remote -v
git branch --show-current
```

You want the top level to be the node folder, not the full ComfyUI folder.

Good:

```text
C:/AI/ComfyUIOpti/ComfyUICuda13/ComfyUI/custom_nodes/mAI_ImageBrowser
```

Bad:

```text
C:/AI/ComfyUIOpti/ComfyUICuda13/ComfyUI
```

The remote should be your repo, not:

```text
https://github.com/Comfy-Org/ComfyUI.git
```

## 6. One-time Git safe directory fix

If Git complains about dubious ownership, run this once:

```powershell
git config --global --add safe.directory "C:/AI/ComfyUIOpti/ComfyUICuda13/ComfyUI/custom_nodes/*"
```

This trusts Git repos directly inside `custom_nodes`.

Avoid this unless you really know why you need it:

```powershell
git config --global --add safe.directory "*"
```

## 7. Connect it to Codex

In ChatGPT, connect GitHub from:

```text
Settings > Apps > GitHub
```

Choose only the repository you want Codex to access.

In Codex, select this repository as the project.

Codex should see this root file:

```text
AGENTS.md
```

That file tells Codex how to work safely in this ComfyUI node pack.

## 8. Recommended Codex workflow

Before asking Codex to work, create a branch:

```powershell
git checkout main
git pull
git checkout -b codex/add-image-browser-node
```

Then ask Codex to work on that branch.

Example prompt:

```text
Implement a ComfyUI custom node pack feature in this repository only.
Follow AGENTS.md strictly.
Do not modify ComfyUI core.
Do not add torch to requirements.txt.
Keep the node small and isolated.
Update README with usage.
Add a small pure Python test if useful.
At the end, explain what changed and how to test it in ComfyUI.
```

After Codex changes files:

```powershell
git status
git diff
```

Test in ComfyUI.

If good:

```powershell
git add .
git commit -m "Add image browser node"
git push -u origin codex/add-image-browser-node
```
