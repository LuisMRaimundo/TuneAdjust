@echo off
title Tune Detection B
cd /d "%~dp0"

python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found.
    echo Install Python 3.10+ then run: pip install -r requirements.txt
    pause
    exit /b 1
)

if /i "%~1"=="menu" goto menu
if /i "%~1"=="cli" goto autocorrect
if /i "%~1"=="tuner" goto livetuner

echo Starting Note Frequency Analyzer...
python note_frequency_analyzer.py
if errorlevel 1 pause
exit /b 0

:menu
cls
echo.
echo  ========================================
echo   Tune Detection B
echo  ========================================
echo.
echo   [1] Note Frequency Analyzer  ^(recommended^)
echo   [2] Auto-correct folder      ^(CLI^)
echo   [2B] Auto-correct all subfolders ^(CLI batch^)
echo   [3] Live Tuner
echo   [Q] Quit
echo.
set /p CHOICE="Choose 1, 2, 2B, 3 or Q [default 1]: "

if "%CHOICE%"=="" set CHOICE=1
if /i "%CHOICE%"=="Q" exit /b 0
if /i "%CHOICE%"=="1" goto analyzer
if /i "%CHOICE%"=="2" goto autocorrect
if /i "%CHOICE%"=="2B" goto autocorrectbatch
if /i "%CHOICE%"=="3" goto livetuner

echo Invalid choice.
pause
goto menu

:analyzer
echo.
echo Starting Note Frequency Analyzer...
python note_frequency_analyzer.py
if errorlevel 1 pause
goto menu

:autocorrect
echo.
set FOLDER=
set /p FOLDER="Folder path with audio files: "
if "%FOLDER%"=="" (
    echo No folder entered.
    pause
    goto menu
)
echo.
echo Dry-run first? [Y/N, default N]
set /p DRY="> "
if /i "%DRY%"=="Y" (
    python auto_correct.py "%FOLDER%" --dry-run
) else (
    python auto_correct.py "%FOLDER%"
)
pause
goto menu

:autocorrectbatch
echo.
set FOLDER=
set /p FOLDER="Parent folder containing subfolders: "
if "%FOLDER%"=="" (
    echo No folder entered.
    pause
    goto menu
)
echo.
echo Dry-run first? [Y/N, default N]
set /p DRY="> "
if /i "%DRY%"=="Y" (
    python auto_correct.py "%FOLDER%" --batch --dry-run
) else (
    python auto_correct.py "%FOLDER%" --batch
)
pause
goto menu

:livetuner
echo.
echo Starting Live Tuner...
python live_tuner.py
if errorlevel 1 pause
goto menu
