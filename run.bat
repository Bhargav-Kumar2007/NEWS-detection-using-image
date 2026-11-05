@echo off
REM Windows runner.   Usage:  run.bat <command> [options]
REM   setup      create .venv and install requirements (CUDA PyTorch if an NVIDIA GPU is found)
REM   download   download + preprocess data      (add --full for the full dataset)
REM   train      train (auto-downloads data)     (add --full, --model, --epochs, ...)
REM   evaluate   evaluate best checkpoint        (--split val/test, --checkpoint ...)
REM   predict    predict images                  (run.bat predict img1.jpg img2.png)
REM   all        setup (if needed) + download + train + evaluate
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "CMD=%~1"
if "%CMD%"=="" set "CMD=help"

set "ARGS="
set "FULL="
:collect
shift
if "%~1"=="" goto collected
set "ARGS=!ARGS! %1"
if /I "%~1"=="--full" set "FULL=--full"
goto collect
:collected

set "VPY=.venv\Scripts\python.exe"

if /I "%CMD%"=="setup" goto setup
if /I "%CMD%"=="download" goto download
if /I "%CMD%"=="train" goto train
if /I "%CMD%"=="evaluate" goto evaluate
if /I "%CMD%"=="predict" goto predict
if /I "%CMD%"=="all" goto all
goto usage

:ensure_env
if exist "%VPY%" goto :eof
call :setup_env
goto :eof

:setup_env
where py >nul 2>nul
if %errorlevel%==0 ( py -3 -m venv .venv ) else ( python -m venv .venv )
if errorlevel 1 ( echo Failed to create virtual environment. Is Python 3.9+ installed? & exit /b 1 )
"%VPY%" -m pip install --upgrade pip
where nvidia-smi >nul 2>nul
if %errorlevel%==0 (
  echo NVIDIA GPU detected - installing CUDA build of PyTorch...
  "%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)
"%VPY%" -m pip install -r requirements.txt
echo Setup complete.
goto :eof

:setup
call :setup_env
goto end

:download
call :ensure_env
"%VPY%" download_data.py !ARGS!
goto end

:train
call :ensure_env
"%VPY%" train.py !ARGS!
goto end

:evaluate
call :ensure_env
"%VPY%" evaluate.py !ARGS!
goto end

:predict
call :ensure_env
"%VPY%" predict.py !ARGS!
goto end

:all
call :ensure_env
"%VPY%" download_data.py !FULL!
if errorlevel 1 goto end
"%VPY%" train.py !ARGS!
goto end

:usage
echo Usage: run.bat ^<setup^|download^|train^|evaluate^|predict^|all^> [options]
echo Add --full to use the full dataset.

:end
endlocal
