@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ------------------------------------------------------------
REM mAI ComfyUI custom node Git + Codex setup helper
REM Run this file from inside the copied and renamed node folder.
REM
REM What it does:
REM - Checks Git
REM - Adds this folder as a Git safe.directory
REM - Initializes Git
REM - Creates main branch
REM - Creates first commit
REM - Optionally creates the GitHub repo using GitHub CLI gh
REM - Adds origin and pushes main
REM ------------------------------------------------------------

title mAI ComfyUI Node Git + GitHub Setup

where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git was not found in PATH.
    echo Install Git for Windows first, then run this file again.
    echo https://git-scm.com/download/win
    pause
    exit /b 1
)

cd /d "%~dp0"
set "REPO_ROOT=%CD%"
for %%I in ("%REPO_ROOT%") do set "DEFAULT_REPO_NAME=%%~nxI"

echo.
echo ============================================================
echo  mAI ComfyUI custom node Git + GitHub setup
echo ============================================================
echo.
echo Repo folder:
echo %REPO_ROOT%
echo.

REM Add current folder as safe Git directory, useful when Windows ownership differs.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=(Resolve-Path '.').Path -replace '\\','/'; git config --global --add safe.directory $p" >nul 2>nul

REM If this folder is directly inside custom_nodes, trust all direct child repos there too.
for %%P in ("%REPO_ROOT%\..") do set "PARENT_DIR=%%~fP"
for %%N in ("%PARENT_DIR%") do set "PARENT_NAME=%%~nxN"
if /I "%PARENT_NAME%"=="custom_nodes" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=(Resolve-Path '..').Path -replace '\\','/'; git config --global --add safe.directory ($p + '/*')" >nul 2>nul
)

echo [1/7] Checking Git identity...
set "GIT_NAME="
set "GIT_EMAIL="
for /f "delims=" %%A in ('git config --global user.name 2^>nul') do set "GIT_NAME=%%A"
for /f "delims=" %%A in ('git config --global user.email 2^>nul') do set "GIT_EMAIL=%%A"

if "%GIT_NAME%"=="" (
    echo Git user.name is not set.
    set /p GIT_NAME_INPUT="Your Git name, for example Mathieu Zylberait: "
    if not "!GIT_NAME_INPUT!"=="" git config --global user.name "!GIT_NAME_INPUT!"
)

if "%GIT_EMAIL%"=="" (
    echo Git user.email is not set.
    set /p GIT_EMAIL_INPUT="Your Git email, for example you@example.com: "
    if not "!GIT_EMAIL_INPUT!"=="" git config --global user.email "!GIT_EMAIL_INPUT!"
)

echo.
echo [2/7] Initializing local repository...
if not exist ".git" (
    git init
    if errorlevel 1 goto :git_error
) else (
    echo Git repo already exists.
)

echo.
echo [3/7] Setting branch to main...
git branch -M main
if errorlevel 1 goto :git_error

echo.
echo [4/7] Creating initial commit if needed...
git status --porcelain > "%TEMP%\mai_git_status.txt"
for %%F in ("%TEMP%\mai_git_status.txt") do set STATUS_SIZE=%%~zF
if %STATUS_SIZE% GTR 0 (
    git add .
    if errorlevel 1 goto :git_error
    git commit -m "Initial ComfyUI custom node pack"
    if errorlevel 1 goto :git_error
) else (
    echo No changes to commit.
)
del "%TEMP%\mai_git_status.txt" >nul 2>nul

echo.
echo [5/7] GitHub repo creation option...
echo This script can create the GitHub repo for you using GitHub CLI, called gh.
echo It will use the URL created by GitHub as the origin remote.
echo.
set "AUTO_CREATE=Y"
set /p AUTO_CREATE="Create GitHub repo automatically with gh? [Y/n]: "
if "%AUTO_CREATE%"=="" set "AUTO_CREATE=Y"
if /I "%AUTO_CREATE%"=="Y" goto :auto_create_repo
if /I "%AUTO_CREATE%"=="YES" goto :auto_create_repo

goto :manual_remote

:auto_create_repo
echo.
echo [6/7] Checking GitHub CLI...
where gh >nul 2>nul
if errorlevel 1 (
    echo [ERROR] GitHub CLI was not found.
    echo.
    echo Install it with one of these options:
    echo winget install --id GitHub.cli
    echo or download it from https://cli.github.com/
    echo.
    echo Falling back to manual remote URL setup.
    goto :manual_remote
)

gh auth status >nul 2>nul
if errorlevel 1 (
    echo You are not logged in to GitHub CLI yet.
    echo A browser login will open now.
    gh auth login
    if errorlevel 1 goto :gh_auth_error
)

echo.
echo Repo name default: %DEFAULT_REPO_NAME%
set "GH_REPO_NAME="
set /p GH_REPO_NAME="GitHub repo name, press Enter for default: "
if "%GH_REPO_NAME%"=="" set "GH_REPO_NAME=%DEFAULT_REPO_NAME%"

echo.
echo Visibility default: private
set "GH_VISIBILITY=private"
set /p GH_VISIBILITY="Visibility [private/public]: "
if "%GH_VISIBILITY%"=="" set "GH_VISIBILITY=private"
if /I "%GH_VISIBILITY%"=="public" (
    set "GH_VISIBILITY_FLAG=--public"
) else (
    set "GH_VISIBILITY_FLAG=--private"
)

echo.
echo Optional: create under a GitHub organization or specific owner.
echo Leave empty to create under your personal GitHub account.
set "GH_OWNER="
set /p GH_OWNER="Owner or org, optional: "
if "%GH_OWNER%"=="" (
    set "GH_FULL_REPO=%GH_REPO_NAME%"
) else (
    set "GH_FULL_REPO=%GH_OWNER%/%GH_REPO_NAME%"
)

echo.
echo About to create:
echo %GH_FULL_REPO% %GH_VISIBILITY_FLAG%
echo.
set "CONFIRM_CREATE=Y"
set /p CONFIRM_CREATE="Continue? [Y/n]: "
if "%CONFIRM_CREATE%"=="" set "CONFIRM_CREATE=Y"
if /I not "%CONFIRM_CREATE%"=="Y" if /I not "%CONFIRM_CREATE%"=="YES" goto :manual_remote

git remote get-url origin >nul 2>nul
if not errorlevel 1 (
    echo.
    echo A remote named origin already exists:
    git remote -v
    echo.
    set "OVERWRITE_ORIGIN="
    set /p OVERWRITE_ORIGIN="Remove origin and let gh recreate it? [y/N]: "
    if /I "!OVERWRITE_ORIGIN!"=="Y" (
        git remote remove origin
        if errorlevel 1 goto :git_error
    ) else (
        echo Keeping existing origin. Trying to push current main branch.
        git push -u origin main
        if errorlevel 1 goto :push_error
        goto :done
    )
)

echo.
echo [7/7] Creating GitHub repo and pushing...
gh repo create "%GH_FULL_REPO%" %GH_VISIBILITY_FLAG% --source . --remote origin --push
if errorlevel 1 goto :gh_create_error

goto :done

:manual_remote
echo.
echo [6/7] Manual GitHub remote setup...
echo Create an EMPTY GitHub repo manually, without README or gitignore.
echo Example: https://github.com/YOUR_USERNAME/YOUR_REPO.git
echo.
set "REMOTE_URL="
set /p REMOTE_URL="Paste remote URL now, or press Enter to skip: "

if not "%REMOTE_URL%"=="" (
    git remote get-url origin >nul 2>nul
    if errorlevel 1 (
        git remote add origin "%REMOTE_URL%"
    ) else (
        git remote set-url origin "%REMOTE_URL%"
    )
    if errorlevel 1 goto :git_error

    echo.
    echo [7/7] Pushing to GitHub...
    git push -u origin main
    if errorlevel 1 goto :push_error
) else (
    echo.
    echo Skipped remote setup.
    echo Later you can run:
    echo git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
    echo git push -u origin main
)

goto :done

:done
echo.
echo ============================================================
echo  Done.
echo ============================================================
echo.
echo Current Git remote:
git remote -v
 echo.
echo Next steps:
echo 1. Connect this GitHub repo in ChatGPT: Settings ^> Apps ^> GitHub
echo 2. Open Codex and select this repo
echo 3. Codex will read AGENTS.md for project rules
echo.
echo Recommended first Codex branch:
echo git checkout -b codex/first-node-feature
echo.
pause
exit /b 0

:git_error
echo.
echo [ERROR] Git setup failed.
echo Run these commands manually to inspect the problem:
echo git status
echo git remote -v
echo git config --global --get-all safe.directory
echo.
pause
exit /b 1

:push_error
echo.
echo [ERROR] Push failed.
echo Most common causes:
echo - The GitHub repo does not exist yet.
echo - The remote URL is wrong.
echo - You are not logged in to GitHub from Git.
echo - The remote repo is not empty.
echo.
echo Check with:
echo git remote -v
echo git status
echo.
pause
exit /b 1

:gh_auth_error
echo.
echo [ERROR] GitHub CLI login failed.
echo Try manually:
echo gh auth login
echo gh auth status
echo.
pause
exit /b 1

:gh_create_error
echo.
echo [ERROR] GitHub repo creation failed.
echo Most common causes:
echo - The repo already exists.
echo - You do not have permission to create repos under that owner/org.
echo - GitHub CLI is logged into the wrong account.
echo - Network/authentication issue.
echo.
echo Check with:
echo gh auth status
echo git remote -v
echo.
echo You can still create the repo manually on GitHub and run this script again.
echo.
pause
exit /b 1
