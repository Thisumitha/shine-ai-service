<#
  One-command start for the Shine AI service (like `npm install && npm start`).

    .\start.ps1                 set everything up, then run the server
    .\start.ps1 -SetupOnly      set up but do not start the server
    .\start.ps1 -SkipOcrSetup   do not install Tesseract or download language data
    .\start.ps1 -Port 9000

  What it does: creates .venv, installs requirements.txt, copies .env.example to .env,
  installs Tesseract with winget if missing, downloads OCR language data into data\tessdata,
  then starts uvicorn. MySQL must already be running (the database is created if missing).
#>
param(
    [int]$Port = 0,
    [switch]$SetupOnly,
    [switch]$SkipOcrSetup,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Warn($text) { Write-Host "    ! $text" -ForegroundColor Yellow }

function Get-EnvValue($name) {
    if (-not (Test-Path .env)) { return "" }
    $line = Get-Content .env | Where-Object { $_ -match "^\s*$name\s*=" } | Select-Object -First 1
    if (-not $line) { return "" }
    return ($line -split "=", 2)[1].Trim()
}

function Set-EnvValue($name, $value) {
    $lines = @(Get-Content .env)
    $found = $false
    $lines = $lines | ForEach-Object {
        if ($_ -match "^\s*$name\s*=") { $found = $true; "$name=$value" } else { $_ }
    }
    if (-not $found) { $lines += "$name=$value" }
    [System.IO.File]::WriteAllLines((Join-Path $PSScriptRoot ".env"), [string[]]$lines)  # UTF-8 without BOM
}

# 1. virtual environment and packages
Step "Python environment"
if (-not (Test-Path .venv\Scripts\python.exe)) {
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python is not installed or not on PATH." }
    python -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --quiet --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
Write-Host "    packages are up to date"

# 2. .env
Step "Settings (.env)"
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Warn ".env was created from .env.example. Fill in INTERNAL_SECRET, JWT_SECRET and DATABASE_URL, then run again."
    if (-not $SetupOnly) { exit 1 }
}
if (-not (Get-EnvValue "INTERNAL_SECRET")) { Warn "INTERNAL_SECRET is empty, so /internal/* endpoints will answer 401." }

# 3. Tesseract and language data
if (-not $SkipOcrSetup) {
    Step "OCR (Tesseract)"
    $tess = Get-EnvValue "TESSERACT_CMD"
    if (-not $tess -or -not (Test-Path $tess)) {
        $onPath = Get-Command tesseract -ErrorAction SilentlyContinue
        $default = "C:\Program Files\Tesseract-OCR\tesseract.exe"
        if ($onPath) { $tess = $onPath.Source }
        elseif (Test-Path $default) { $tess = $default }
        elseif (Get-Command winget -ErrorAction SilentlyContinue) {
            Write-Host "    installing Tesseract with winget (a permission prompt may appear)"
            winget install --id UB-Mannheim.TesseractOCR -e --accept-package-agreements --accept-source-agreements
            if (Test-Path $default) { $tess = $default }
        }
    }
    if ($tess -and (Test-Path $tess)) {
        Write-Host "    found $tess"
        if ((Get-EnvValue "TESSERACT_CMD") -ne $tess) { Set-EnvValue "TESSERACT_CMD" $tess }
    }
    else {
        Warn "Tesseract not found. Scanned pages will come back empty. Install it from the UB Mannheim page, then run again."
    }

    $languages = Get-EnvValue "OCR_LANGUAGE"
    if (-not $languages) { $languages = "eng" }
    $tessdata = Join-Path $PSScriptRoot "data\tessdata"
    New-Item -ItemType Directory -Force $tessdata | Out-Null
    foreach ($code in ($languages -split "\+")) {
        $file = Join-Path $tessdata "$code.traineddata"
        if (Test-Path $file) { continue }
        Write-Host "    downloading $code language data"
        try {
            Invoke-WebRequest "https://github.com/tesseract-ocr/tessdata_fast/raw/main/$code.traineddata" -OutFile $file
        }
        catch {
            Remove-Item $file -ErrorAction SilentlyContinue
            Warn "could not download '$code' ($($_.Exception.Message))"
        }
    }
    $env:TESSDATA_PREFIX = $tessdata
}

# 4. database
Step "Database"
& $py -c @"
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from app.config import get_settings
url = make_url(get_settings().database_url)
if not url.drivername.startswith('mysql'):
    sys.exit(0)
try:
    admin = create_engine(url.set(database=None))
    with admin.connect() as db:
        db.execute(text('CREATE DATABASE IF NOT EXISTS ``%s`` CHARACTER SET utf8mb4' % url.database))
        db.commit()
    print('    database', url.database, 'is ready')
except Exception as error:
    print('    ! cannot reach MySQL:', error)
    sys.exit(1)
"@
if ($LASTEXITCODE -ne 0) { throw "MySQL is not reachable. Start MySQL and check DATABASE_URL in .env." }

if ($SetupOnly) { Write-Host "`nSetup finished." -ForegroundColor Green; exit 0 }

# 5. run
if ($Port -eq 0) {
    $fromEnv = Get-EnvValue "PORT"
    $Port = if ($fromEnv) { [int]$fromEnv } else { 8000 }
}
Step "Starting on http://localhost:$Port  (docs: /docs)"
$uvicornArgs = @("-m", "uvicorn", "app.main:app", "--port", "$Port")
if (-not $NoReload) { $uvicornArgs += "--reload" }
& $py @uvicornArgs
