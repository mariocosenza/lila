#!/bin/bash

# ============================================
# Script per avviare tutti i servizi LILA
# Avvia contemporaneamente:
# 1. Server MCP (grammo-mcp) - porta 8000
# 2. Tester Server - porta 8088
# 3. Agent Client (REPL interattivo)
# ============================================

# Configurazione
GEMINI_MODEL="${1:-gemini-3-27b-it}"
MCP_PORT="${2:-8000}"
TESTER_PORT="${3:-8088}"

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Funzione per output colorato
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verifica che siamo nella directory corretta
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
log_info "Working directory: $SCRIPT_DIR"

# ============================================
# 1. Avvia Server MCP (porta 8000)
# ============================================
log_info "Starting MCP Server on port $MCP_PORT..."
python mcp/server.py &
MCP_PID=$!
log_success "MCP Server started with PID: $MCP_PID"

# ============================================
# 2. Avvia Tester Server (porta 8088)
# ============================================
log_info "Starting Tester Server on port $TESTER_PORT..."
log_info "Using Gemini model: $GEMINI_MODEL"

export GEMINI_MODEL=$GEMINI_MODEL
python agents/tester_server.py &
TESTER_PID=$!
log_success "Tester Server started with PID: $TESTER_PID"

# ============================================
# 3. Avvia Agent Client (REPL interattivo)
# ============================================
log_info "Starting Agent Client (REPL)..."
log_warning "Waiting 3 seconds for services to initialize..."
sleep 3

log_success "Launching Agent Client..."
python agents/agent_client.py

# ============================================
# Pulizia
# ============================================
log_warning "Shutting down all services..."
kill $MCP_PID 2>/dev/null || true
kill $TESTER_PID 2>/dev/null || true
log_success "All services stopped"
