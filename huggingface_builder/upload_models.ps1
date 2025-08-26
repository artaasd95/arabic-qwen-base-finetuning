# Hugging Face Model Uploader - PowerShell Script
# This script sets up the environment and runs the model uploader

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Hugging Face Model Repository Builder" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if a command exists
function Test-Command($cmdname) {
    return [bool](Get-Command -Name $cmdname -ErrorAction SilentlyContinue)
}

# Function to check if a Python package is installed
function Test-PythonPackage($packagename) {
    try {
        $result = python -c "import $packagename" 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

# Check if Python is installed
if (-not (Test-Command "python")) {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and add it to your PATH" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Python found: " -NoNewline
python --version
Write-Host ""

# Check if we're in the correct directory
if (-not (Test-Path "upload_models.py")) {
    Write-Host "ERROR: upload_models.py not found" -ForegroundColor Red
    Write-Host "Please run this script from the huggingface_builder directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "WARNING: .env file not found" -ForegroundColor Yellow
    Write-Host "Please copy .env.example to .env and configure your credentials" -ForegroundColor Yellow
    Write-Host ""
    $choice = Read-Host "Would you like to copy .env.example to .env now? (y/n)"
    
    if ($choice -eq "y" -or $choice -eq "Y") {
        Copy-Item ".env.example" ".env"
        Write-Host ".env file created. Please edit it with your credentials." -ForegroundColor Green
        Write-Host "Opening .env file in notepad..." -ForegroundColor Cyan
        
        # Try to open with VS Code first, then notepad
        if (Test-Command "code") {
            code .env
        } else {
            notepad .env
        }
        
        Write-Host ""
        Write-Host "Please save the file and run this script again." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 0
    } else {
        Write-Host "Please create .env file manually and run this script again." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check if requirements are installed
Write-Host "Checking dependencies..." -ForegroundColor Cyan

if (-not (Test-PythonPackage "huggingface_hub")) {
    Write-Host "Installing required dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
        Write-Host "Please check your internet connection and try again" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "Dependencies already installed." -ForegroundColor Green
}
Write-Host ""

# Check if model data exists
if (-not (Test-Path "..\outputs\model_paths.json")) {
    Write-Host "ERROR: Model data not found" -ForegroundColor Red
    Write-Host "Please ensure you have trained models and outputs/model_paths.json exists" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path "..\outputs\training_results.json")) {
    Write-Host "ERROR: Training results not found" -ForegroundColor Red
    Write-Host "Please ensure you have training_results.json in the outputs directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Model data found. Starting upload process..." -ForegroundColor Green
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IMPORTANT: This will upload models to Hugging Face" -ForegroundColor Yellow
Write-Host "Make sure your .env file is configured correctly" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Show .env file contents (masked)
Write-Host "Current configuration:" -ForegroundColor Cyan
if (Test-Path ".env") {
    $envContent = Get-Content ".env"
    foreach ($line in $envContent) {
        if ($line -match "^HUGGINGFACE_TOKEN=(.+)$") {
            $token = $matches[1]
            if ($token -ne "your_huggingface_token_here" -and $token.Length -gt 10) {
                Write-Host "  HUGGINGFACE_TOKEN=hf_****...****" -ForegroundColor Green
            } else {
                Write-Host "  HUGGINGFACE_TOKEN=<not configured>" -ForegroundColor Red
            }
        }
        elseif ($line -match "^HUGGINGFACE_USERNAME=(.+)$") {
            $username = $matches[1]
            if ($username -ne "your_username_here") {
                Write-Host "  HUGGINGFACE_USERNAME=$username" -ForegroundColor Green
            } else {
                Write-Host "  HUGGINGFACE_USERNAME=<not configured>" -ForegroundColor Red
            }
        }
        elseif ($line -match "^BASE_MODEL_NAME=(.+)$") {
            Write-Host "  BASE_MODEL_NAME=$($matches[1])" -ForegroundColor Cyan
        }
    }
}
Write-Host ""

$continue = Read-Host "Press Enter to continue or Ctrl+C to cancel"
Write-Host ""

# Run the uploader
Write-Host "Starting Hugging Face model upload..." -ForegroundColor Cyan
Write-Host ""

try {
    python upload_models.py
    $exitCode = $LASTEXITCODE
} catch {
    Write-Host "ERROR: Failed to run upload script" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    $exitCode = 1
}

# Check if upload was successful
Write-Host ""
if ($exitCode -ne 0) {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Upload completed with errors" -ForegroundColor Red
    Write-Host "Please check the log messages above" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Red
} else {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Upload completed successfully!" -ForegroundColor Green
    Write-Host "Check your Hugging Face profile for the new repositories" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Green
}

Write-Host ""
Read-Host "Press Enter to exit"