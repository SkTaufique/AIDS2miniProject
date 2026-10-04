import os
import sys
import time
import webbrowser
import threading

def launch_browser():
    time.sleep(1.8)
    print("\n[LAUNCH] Opening Veritas Multimodal Dashboard in browser...")
    webbrowser.open("http://127.0.0.1:8000")

def main():
    print("=" * 60)
    print("  VERITAS AI — Multimodal Fake Content Detection System")
    print("  Semester 7 AI & Data Science Project")
    print("=" * 60)

    # Check CUDA GPU status
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            print(f"[HARDWARE] Hardware Acceleration: CUDA ENABLED ({gpu})")
        else:
            print("[HARDWARE] Running on CPU Mode")
    except Exception as e:
        print(f"[HARDWARE] Notice: {e}")

    # Launch browser thread
    threading.Thread(target=launch_browser, daemon=True).start()

    # Start FastAPI / Uvicorn Server
    import uvicorn
    print("\n[SERVER] Starting API server on http://127.0.0.1:8000 ...")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
