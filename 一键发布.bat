@echo off
:: ?? 这一行是关键！强制跳转到脚本所在的目录
cd /d "%~dp0"

echo ==========================================
echo       Auto Syncing to Cloud...
echo ==========================================
echo Working Directory: %cd%
echo.

:: 1. Add files
echo [1/3] Adding files (git add)...
git add .

:: 2. Commit
echo [2/3] Committing (git commit)...
set mydate=%date:~0,10%
set mytime=%time:~0,8%
git commit -m "Auto update: %mydate% %mytime%"

:: 3. Push
echo [3/3] Pushing to GitHub (git push)...
git push

echo.
echo ==========================================
echo    Success! Please wait for Vercel update.
echo ==========================================
pause