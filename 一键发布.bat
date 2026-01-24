@echo off
chcp 65001 >nul
echo ==========================================
echo       正在为你自动同步到云端...
echo ==========================================

:: 1. 添加所有文件
echo [1/3] 正在装箱 (git add)...
git add .

:: 2. 提交更改 (自动加上时间戳)
echo [2/3] 正在贴单 (git commit)...
set mydate=%date:~0,10%
set mytime=%time:~0,8%
git commit -m "Auto update: %mydate% %mytime%"

:: 3. 推送到 GitHub
echo [3/3] 正在发货 (git push)...
git push

echo.
echo ==========================================
echo    ? 成功！Vercel 正在构建，请稍等1分钟。
echo ==========================================
pause