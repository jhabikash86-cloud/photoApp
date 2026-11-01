#!/bin/bash
# Neural Edit Pro launcher: starts the editor's two servers if they are not already running, then opens it in its own app window.
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
PROJECT="/Users/gaurisusheela/Downloads/my-ai-photo-app"
LOG="$HOME/Library/Logs/NeuralEditPro"
mkdir -p "$LOG"
up() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
cd "$PROJECT" || { osascript -e 'display alert "Neural Edit Pro" message "The project folder was moved or deleted."'; exit 1; }

if ! up 8000; then
  nohup ./venv/bin/uvicorn server:app --port 8000 >> "$LOG/backend.log" 2>&1 &
fi
if ! up 3000; then
  [ -d pro-editor-frontend/.next-prod ] || (cd pro-editor-frontend && NEP_PROD=1 npm run build >> "$LOG/build.log" 2>&1)
  (cd pro-editor-frontend && NEP_PROD=1 nohup npm run start -- -p 3000 >> "$LOG/frontend.log" 2>&1 &)
fi
for i in $(seq 1 120); do up 8000 && up 3000 && break; sleep 1; done
if ! up 8000 || ! up 3000; then
  osascript -e 'display alert "Neural Edit Pro could not start" message "See ~/Library/Logs/NeuralEditPro for details."'; exit 1
fi
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ -x "$CHROME" ]; then
  nohup "$CHROME" --app=http://localhost:3000 --window-size=1500,950 >/dev/null 2>&1 &
  disown
else
  open "http://localhost:3000"
fi
