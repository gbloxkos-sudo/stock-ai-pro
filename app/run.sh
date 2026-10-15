#!/bin/bash
echo "Starting StockAI Dashboard..."
echo "Ollama status: $(curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('Connected -' + str(len(d.get('models',[]))) + ' models')" 2>/dev/null || echo 'Not running')"
echo "Launching on http://localhost:8501"
export PATH="$HOME/.local/bin:$PATH"
streamlit run app.py