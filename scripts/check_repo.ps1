$ErrorActionPreference = "Stop"

Write-Host "Current folder:"
pwd

Write-Host "Git top level:"
git rev-parse --show-toplevel

Write-Host "Git remotes:"
git remote -v

Write-Host "Current branch:"
git branch --show-current
