# ============================================
# Script per avviare tutti i servizi LILA
# Avvia contemporaneamente:
# 1. Server MCP (grammo-mcp) - porta 8000
# 2. Tester Server - porta 8088
# 3. Agent Client (REPL interattivo)
# ============================================

param(
    [int]$McpPort = 8000,
    [int]$TesterPort = 8088,
    [string]$GeminiModel = "gemma-3-27b-it",
    [switch]$Local = $false,
    [string]$LocalModel = "gpt-oss:20b"
)

# Colori per output
$colors = @{
    Info = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Type = "Info"
    )
    Write-Host $Message -ForegroundColor $colors[$Type]
}

# Verifica che siamo nella directory corretta
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir
Write-ColorOutput "[ROOT] Working directory: $RootDir" "Info"

# ============================================
# Configurazione LLM
# ============================================
if ($Local) {
    $env:USE_LOCAL_LLM = "true"
    $env:OLLAMA_MODEL = $LocalModel
    Write-ColorOutput "[CONFIG] Using LOCAL LLM (Ollama): $LocalModel" "Warning"
} else {
    $env:USE_LOCAL_LLM = "false"
    $env:GEMINI_MODEL = $GeminiModel
    Write-ColorOutput "[CONFIG] Using GEMINI LLM: $GeminiModel" "Info"
}

# ============================================
# 1. Avvia Server MCP (porta 8000)
# ============================================
Write-ColorOutput "`n[MCP] Starting MCP Server on port $McpPort..." "Info"
$McpProcess = Start-Process -NoNewWindow -PassThru -FilePath "python" `
    -ArgumentList "mcp/server.py" `
    -WorkingDirectory $RootDir

Write-ColorOutput "[MCP] Process ID: $($McpProcess.Id)" "Success"

# ============================================
# 2. Avvia Tester Server (porta 8088)
# ============================================
Write-ColorOutput "`n[TESTER] Starting Tester Server on port $TesterPort..." "Info"

$TesterProcess = Start-Process -NoNewWindow -PassThru -FilePath "python" `
    -ArgumentList "agents/tester_server.py" `
    -WorkingDirectory $RootDir

Write-ColorOutput "[TESTER] Process ID: $($TesterProcess.Id)" "Success"

# ============================================
# 3. Avvia Agent Client (REPL interattivo)
# ============================================
Write-ColorOutput "`n[CLIENT] Starting Agent Client (REPL)..." "Info"
Write-ColorOutput "[CLIENT] This will run in the current terminal" "Info"

# Aspetta un momento per far avviare gli altri servizi
Write-ColorOutput "`n[STARTUP] Waiting 3 seconds for services to initialize..." "Warning"
Start-Sleep -Seconds 3

# Avvia il client nello stesso terminal
Write-ColorOutput "`n[CLIENT] Launching Agent Client..." "Success"
python agents/agent_client.py

# ============================================
# Pulizia
# ============================================
Write-ColorOutput "`n[CLEANUP] Shutting down all services..." "Warning"
Stop-Process -Id $McpProcess.Id -Force -ErrorAction SilentlyContinue
Stop-Process -Id $TesterProcess.Id -Force -ErrorAction SilentlyContinue
Write-ColorOutput "[CLEANUP] All services stopped" "Success"
