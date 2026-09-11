# MySQL Password Reset Script
# Run this in PowerShell as Administrator

Write-Host "=== MySQL Password Reset ===" -ForegroundColor Cyan
Write-Host ""

$mysqld = "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysqld.exe"
$mysql  = "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe"
$mycnf  = "C:\ProgramData\MySQL\MySQL Server 8.0\my.ini"

# Step 1: Stop MySQL service
Write-Host "[1/6] Stopping MySQL service..." -ForegroundColor Yellow
net stop MySQL80 2>$null
Start-Sleep -Seconds 3

# Kill any leftover mysqld processes
Write-Host "[2/6] Killing any leftover mysqld processes..." -ForegroundColor Yellow
Stop-Process -Name mysqld -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# Step 3: Create an init file with the password reset SQL
Write-Host "[3/6] Creating password reset init file..." -ForegroundColor Yellow
$initFile = "$env:TEMP\mysql_reset_init.sql"
@"
FLUSH PRIVILEGES;
ALTER USER 'root'@'localhost' IDENTIFIED BY 'akshit1915';
FLUSH PRIVILEGES;
"@ | Out-File -FilePath $initFile -Encoding ASCII -NoNewline

# Step 4: Start MySQL in safe mode with init file
Write-Host "[4/6] Starting MySQL in safe mode (this may take a moment)..." -ForegroundColor Yellow
Start-Process -FilePath $mysqld -ArgumentList "--defaults-file=`"$mycnf`"", "--skip-grant-tables", "--init-file=`"$initFile`"" -WindowStyle Hidden

# Wait and retry connection to confirm it started
$connected = $false
for ($i = 1; $i -le 12; $i++) {
    Start-Sleep -Seconds 5
    Write-Host "       Waiting for MySQL to start... attempt $i/12" -ForegroundColor Gray
    $result = & $mysql -u root -e "SELECT 1;" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "       MySQL is up!" -ForegroundColor Green
        $connected = $true
        break
    }
}

if (-not $connected) {
    Write-Host "ERROR: MySQL did not start in safe mode. Try rebooting and running again." -ForegroundColor Red
    Remove-Item $initFile -ErrorAction SilentlyContinue
    Read-Host "Press Enter to exit"
    exit 1
}

# Step 5: Kill safe-mode MySQL and restart normally
Write-Host "[5/6] Stopping safe-mode MySQL..." -ForegroundColor Yellow
Stop-Process -Name mysqld -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 5

Write-Host "[6/6] Starting MySQL service normally..." -ForegroundColor Yellow
net start MySQL80
Start-Sleep -Seconds 3

# Clean up
Remove-Item $initFile -ErrorAction SilentlyContinue

# Verify
Write-Host ""
Write-Host "=== Verifying connection ===" -ForegroundColor Cyan
& $mysql -u root -pakshit1915 -e "SELECT 'SUCCESS! MySQL password reset complete.' AS Status;" 2>&1

Write-Host ""
if ($LASTEXITCODE -eq 0) {
    Write-Host "Password reset SUCCEEDED! Root password is now: akshit1915" -ForegroundColor Green
} else {
    Write-Host "Password reset may have failed. Try running the script again." -ForegroundColor Red
}

Read-Host "Press Enter to exit"
