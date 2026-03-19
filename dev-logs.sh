#!/usr/bin/env bash

LOG_DIR="/tmp/project-atlas-logs"
SERVICE="${1:-all}"

case "$SERVICE" in
  ui)
    tail -f "$LOG_DIR/ui.log"
    ;;
  api)
    tail -f "$LOG_DIR/api.log"
    ;;
  docker)
    tail -f "$LOG_DIR/docker-compose.log"
    ;;
  all)
    echo "Usage: ./dev-logs.sh [ui|api|docker]"
    echo ""
    echo "Log files:"
    ls -la "$LOG_DIR/"*.log 2>/dev/null || echo "  No logs yet. Run ./dev-start.sh first."
    ;;
  *)
    echo "Unknown service: $SERVICE"
    echo "Usage: ./dev-logs.sh [ui|api|docker]"
    ;;
esac
