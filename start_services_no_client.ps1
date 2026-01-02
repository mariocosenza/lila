# ============================================
# Script per avviare i servizi LILA (senza Client)
# Avvia contemporaneamente:
# 1. Server MCP (grammo-mcp) - porta 8000
# 2. Tester Server - porta 8088
# ============================================

param(
    [int]$McpPort = 8000,
    [int]$TesterPort = 8088,
    [switch]$UseLocalLLM,
    [switch]$Local
)

# Configura l'ambiente per il modello
if ($UseLocalLLM -or $Local) {
    $env:USE_LOCAL_LLM = "true"
    Write-Host "[CONFIG] Using Local LLM (Ollama)" -ForegroundColor Cyan
} else {
    $env:USE_LOCAL_LLM = "false"
    Write-Host "[CONFIG] Using Gemini LLM" -ForegroundColor Cyan
}

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
# Loop di attesa
# ============================================
Write-ColorOutput "`n[RUNNING] Services are running. Press Ctrl+C to stop." "Success"

try {
    while ($true) {
        Start-Sleep -Seconds 1
        
        # Check if processes are still running
        if ($McpProcess.HasExited) {
            Write-ColorOutput "[MCP] Process exited unexpectedly!" "Error"
            break
        }
        if ($TesterProcess.HasExited) {
            Write-ColorOutput "[TESTER] Process exited unexpectedly!" "Error"
            break
        }
    }
}
finally {
    # ============================================
    # Pulizia
    # ============================================
    Write-ColorOutput "`n[CLEANUP] Shutting down all services..." "Warning"
    Stop-Process -Id $McpProcess.Id -Force -ErrorAction SilentlyContinue
    Stop-Process -Id $TesterProcess.Id -Force -ErrorAction SilentlyContinue
    Write-ColorOutput "[CLEANUP] All services stopped" "Success"
}
