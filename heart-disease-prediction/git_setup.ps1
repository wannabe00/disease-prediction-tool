# git_setup.ps1 - Run this after copying all code
# Initialize Git repository and create initial commit

Write-Host "Initializing Git repository..." -ForegroundColor Green
git init

Write-Host "Adding all files..." -ForegroundColor Green
git add .

Write-Host "Creating initial commit..." -ForegroundColor Green
git commit -m "initial: complete project structure and placeholder files"

Write-Host "Git repository initialized successfully!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Create GitHub repository" -ForegroundColor White
Write-Host "2. git remote add origin <your-repo-url>" -ForegroundColor White
Write-Host "3. git push -u origin main" -ForegroundColor White
