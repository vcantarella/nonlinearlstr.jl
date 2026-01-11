$ErrorActionPreference = "Stop"

function Get-PythonPath {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        # Check if it's the Windows Store shim (0-byte file or execution fails)
        if ($cmd.Source -like "*WindowsApps*python.exe") {
            try {
                # Try running version check. If it's the inactive shim, this usually opens the store or fails.
                # We pipe stderr because the shim might write there.
                $process = Start-Process -FilePath "python" -ArgumentList "--version" -NoNewWindow -PassThru -Wait -RedirectStandardError "$env:TEMP\pycheck.err"
                if ($process.ExitCode -eq 0) {
                    return "python"
                }
                return $null
            } catch {
                return $null
            }
        }
        return "python"
    }
    
    # Check common install locations
    $versions = "310", "311", "312", "313"
    foreach ($v in $versions) {
        $path = "$env:LOCALAPPDATA\Programs\Python\Python$v\python.exe"
        if (Test-Path $path) {
            return $path
        }
        $path = "C:\Python$v\python.exe"
        if (Test-Path $path) {
            return $path
        }
    }
    
    return $null
}

$pythonExe = Get-PythonPath

if (-not $pythonExe) {
    Write-Host "Python not found. Attempting to install Python 3.12 via Winget..."
    try {
        winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
        
        # Refresh env vars is hard in current process, so we look for the binary directly
        $stdPath = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
        if (Test-Path $stdPath) {
            $pythonExe = $stdPath
        } else {
             # Try refreshing path for a simple check
             $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
             $pythonExe = Get-PythonPath
        }
        
        if (-not $pythonExe) {
             throw "Python installation appeared to succeed but executable could not be found."
        }
    } catch {
        Write-Error "Python not found and automatic installation failed: $_"
        Write-Error "Please install Python manually (https://www.python.org/downloads/)."
        exit 1
    }
}

Write-Host "Using python: $pythonExe"

# Create venv
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    & $pythonExe -m venv venv
}

# Activate and install
if (Test-Path "venv\Scripts") {
    $pip = "venv\Scripts\pip.exe"
    $venvPython = "$((Get-Location).Path)\venv\Scripts\python.exe"
} else {
    $pip = "venv/bin/pip"
    $venvPython = "$(pwd)/venv/bin/python"
}

Write-Host "Installing dependencies using $pip..."
& $pip install --upgrade pip
& $pip install scipy

Write-Host "Setup complete."
Write-Host "To use with Julia, set: `$env:JULIA_PYTHONCALL_EXE = `"$venvPython`""

