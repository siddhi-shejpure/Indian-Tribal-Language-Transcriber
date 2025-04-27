@echo off
echo ===================================
echo Indian Language Transcriber - Direct Launcher
echo ===================================
echo.

REM Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/downloads/
    exit /b 1
)

REM Create necessary directories
if not exist models mkdir models
if not exist transcriptions mkdir transcriptions

REM Check if dummy model file exists, create if not
if not exist models\tiny.pt (
    echo Creating dummy model file...
    python download_models.py --create-dummy --model tiny
)

REM Install required Python packages if needed
echo Checking and installing required packages...
pip install --no-cache-dir torch==2.0.1+cpu torchaudio==2.0.2+cpu --index-url https://download.pytorch.org/whl/cpu
pip install --no-cache-dir openai-whisper indic-transliteration matplotlib numpy librosa soundfile gradio resampy python-dotenv tqdm requests

REM Run the application with tiny model and local model path
echo Starting application with tiny model...
python mapping.py --web --model tiny --local_model_path models/tiny.pt

echo.
if %ERRORLEVEL% NEQ 0 (
    echo Application exited with an error. Please check the logs above.
) else (
    echo Application finished successfully.
)

pause 