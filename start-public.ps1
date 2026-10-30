# Start SOALM on this PC and expose it publicly via a Cloudflare Quick Tunnel.
# No Cloudflare account needed. The public URL changes every time you run this.
#
# Usage (PowerShell, from this folder):
#   powershell -ExecutionPolicy Bypass -File .\start-public.ps1
#
# Stop: press Ctrl+C in this window (stops the tunnel and the API).

$ErrorActionPreference = "Stop"
$cloudflared = Join-Path $env:USERPROFILE "cloudflared\cloudflared.exe"

if (-not (Test-Path $cloudflared)) {
    Write-Host "Downloading cloudflared..."
    New-Item -ItemType Directory -Force -Path (Split-Path $cloudflared) | Out-Null
    $ProgressPreference = "SilentlyContinue"
    Invoke-WebRequest -UseBasicParsing -OutFile $cloudflared `
        -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
}

# Start the API in the background.
Write-Host "Starting SOALM API (loading 29 models)..."
$api = Start-Process python -ArgumentList "serve.py" -WorkingDirectory $PSScriptRoot -PassThru -NoNewWindow

try {
    # Wait until the models are loaded.
    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $h = Invoke-RestMethod "http://localhost:5000/api/health" -TimeoutSec 3
            if ($h.ready) { $ready = $true; break }
        } catch { }
        Start-Sleep -Seconds 3
    }
    if (-not $ready) { throw "API did not become ready on http://localhost:5000" }
    Write-Host "API ready. Opening public tunnel..."
    Write-Host "Look for the https://....trycloudflare.com URL below."

    & $cloudflared tunnel --no-autoupdate --url http://localhost:5000
}
finally {
    if ($api -and -not $api.HasExited) { Stop-Process -Id $api.Id -Force }
}
