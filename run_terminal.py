"""
FinSight Institutional Equity Research Terminal Runner
Starts the high-performance FastAPI server and opens the Bloomberg-style web desk.
"""

import sys
import os
import webbrowser
import threading
import time
import uvicorn

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:8000")

def main():
    print("=" * 65)
    print("  FINSIGHT | INSTITUTIONAL EQUITY RESEARCH DESK (v2.0)")
    print("=" * 65)
    print("  Status: Starting local server...")
    print("  URL:    http://127.0.0.1:8000")
    print("  Engine: LangGraph Multi-Agent RAG with Deterministic Math Tools")
    print("=" * 65)
    
    # Launch browser automatically
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    main()
