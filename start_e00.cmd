@echo off
cd /d F:\WorkBuddy\workspace\2026-11-16-09-52-10\imagenet-proj
echo [1/3] git add and commit
git add -A
git commit -m "feat: training pipeline with data ready"
echo.
echo [2/3] commit hash - copy into experiment sheet column L
git rev-parse --short HEAD
echo.
echo [3/3] start E00 baseline, about 7.1 hours
call ..\imagenet-schedule\.venv\Scripts\activate.bat
python src\train.py --config configs\e00_baseline.yaml --out runs\E00
echo.
echo done - check runs\E00\summary.json
pause
