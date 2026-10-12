# Ollama setup for the Valorant Voice Coach.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_ollama.ps1
#
# Pulls the default model and configures the two things that actually matter
# for latency on a 6 GB laptop GPU.

$ErrorActionPreference = "Stop"

$Model = "llama3.2:3b-instruct-q4_K_M"

Write-Host "== Checking Ollama ==" -ForegroundColor Cyan
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "Ollama is not installed. Get it from https://ollama.com/download" -ForegroundColor Red
    exit 1
}

# The server must be running before a pull or a run will work.
try {
    Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/version" -UseBasicParsing -TimeoutSec 3 | Out-Null
    Write-Host "Ollama server is running." -ForegroundColor Green
} catch {
    Write-Host "Starting Ollama server..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

Write-Host "`n== Pulling $Model ==" -ForegroundColor Cyan
ollama pull $Model

# Keeping the model resident avoids paying a ~2 s cold load on the first
# question of every session, which is the single most noticeable stall.
Write-Host "`n== Setting OLLAMA_KEEP_ALIVE=60m ==" -ForegroundColor Cyan
[Environment]::SetEnvironmentVariable("OLLAMA_KEEP_ALIVE", "60m", "User")
Write-Host "Set for future sessions. Restart Ollama for it to take effect." -ForegroundColor Green

Write-Host "`n== Verifying GPU placement ==" -ForegroundColor Cyan
ollama run $Model "ok" | Out-Null
ollama ps

Write-Host @"

PROCESSOR must read '100% GPU'. Any CPU share and throughput collapses.

Context length note: Ollama's OpenAI-compatible endpoint ignores a per-request
num_ctx, so LLM_NUM_CTX in .env cannot change it. Context is fixed when the
model loads. To use a different context length, create a variant:

    'FROM $Model'          | Out-File -Encoding ascii Modelfile
    'PARAMETER num_ctx 8192' | Out-File -Encoding ascii -Append Modelfile
    ollama create valcoach:ctx8192 -f Modelfile

then point LLM_MODEL at valcoach:ctx8192.

Measured on an RTX 4050 Laptop (6 GB), Windows taking ~0.8 GB:
  llama3.2:3b-instruct-q4_K_M  2.6 GB  100% GPU   75 tok/s   <- recommended
  llama3.1:8b-instruct-q4_K_M  5.6 GB  78% GPU    ~4-5 s/answer, does not fit

"@ -ForegroundColor Yellow
