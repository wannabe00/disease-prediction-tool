# install_dependencies.ps1 - Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Green

# Check if Python is installed
python --version
if ( -ne 0) {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ first" -ForegroundColor Yellow
    exit 1
}

# Check if pip is available
pip --version
if ( -ne 0) {
    Write-Host "❌ pip is not available" -ForegroundColor Red
    exit 1
}

# Install requirements
Write-Host "Installing requirements from requirements.txt..." -ForegroundColor Green
pip install -r requirements.txt

if ( -eq 0) {
    Write-Host "✅ Dependencies installed successfully!" -ForegroundColor Green
    Write-Host "You can now run: python run.py" -ForegroundColor White
} else {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
}
