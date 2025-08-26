@echo off
REM Hugging Face Model Uploader - Windows Batch Script
REM This script sets up the environment and runs the model uploader

echo ========================================
echo Hugging Face Model Repository Builder
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

REM Check if we're in the correct directory
if not exist "upload_models.py" (
    echo ERROR: upload_models.py not found
    echo Please run this script from the huggingface_builder directory
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo WARNING: .env file not found
    echo Please copy .env.example to .env and configure your credentials
    echo.
    echo Would you like to copy .env.example to .env now? (y/n)
    set /p choice="Enter choice: "
    if /i "%choice%"=="y" (
        copy ".env.example" ".env"
        echo .env file created. Please edit it with your credentials.
        echo Opening .env file in notepad...
        notepad .env
        echo.
        echo Please save the file and run this script again.
        pause
        exit /b 0
    ) else (
        echo Please create .env file manually and run this script again.
        pause
        exit /b 1
    )
)

REM Check if requirements are installed
echo Checking dependencies...
pip show huggingface-hub >nul 2>&1
if errorlevel 1 (
    echo Installing required dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        echo Please check your internet connection and try again
        pause
        exit /b 1
    )
) else (
    echo Dependencies already installed.
)
echo.

REM Check if model data exists
if not exist "..\outputs\model_paths.json" (
    echo ERROR: Model data not found
    echo Please ensure you have trained models and outputs/model_paths.json exists
    pause
    exit /b 1
)

if not exist "..\outputs\training_results.json" (
    echo ERROR: Training results not found
    echo Please ensure you have training_results.json in the outputs directory
    pause
    exit /b 1
)

echo Model data found. Starting upload process...
echo.
echo ========================================
echo IMPORTANT: This will upload models to Hugging Face
echo Make sure your .env file is configured correctly
echo ========================================
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause >nul
echo.

REM Run the uploader
echo Starting Hugging Face model upload...
echo.
python upload_models.py

REM Check if upload was successful
if errorlevel 1 (
    echo.
    echo ========================================
    echo Upload completed with errors
    echo Please check the log messages above
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Upload completed successfully!
    echo Check your Hugging Face profile for the new repositories
    echo ========================================
)

echo.
echo Press any key to exit...
pause >nul