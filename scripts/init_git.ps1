param(
    [string]$RemoteUrl = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

Write-Host "Repo root: $RepoRoot"

$SafePath = ($RepoRoot -replace "\\", "/")
git config --global --add safe.directory $SafePath

if (-not (Test-Path ".git")) {
    git init
}

git branch -M main

$status = git status --porcelain
if ($status) {
    git add .
    git commit -m "Initial ComfyUI custom node pack"
} else {
    Write-Host "No changes to commit."
}

if ($RemoteUrl -ne "") {
    $remotes = git remote
    if ($remotes -contains "origin") {
        git remote set-url origin $RemoteUrl
    } else {
        git remote add origin $RemoteUrl
    }

    git push -u origin main
} else {
    Write-Host "No remote URL provided. Add one later with:"
    Write-Host "git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git"
    Write-Host "git push -u origin main"
}
