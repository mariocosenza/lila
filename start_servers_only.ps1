# Start servers only
$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RootDir

$env:GEMINI_MODEL = "gemma-3-27b-it"

Write-Host "Starting MCP Server..."
$McpProcess = Start-Process -NoNewWindow -PassThru -FilePath "python" -ArgumentList "mcp/server.py"
Write-Host "MCP PID: $($McpProcess.Id)"

Write-Host "Servers running in background."
# Exit script, processes continue running
exit