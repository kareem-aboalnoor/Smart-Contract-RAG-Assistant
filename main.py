"""
main.py — Application Entry Point
Starts the FastAPI server and/or Gradio UI based on command-line arguments.

Usage:
    python main.py              # Launch Gradio UI only
    python main.py --api        # Launch FastAPI server only
    python main.py --both       # Launch both (API + UI)
    python main.py --evaluate   # Run evaluation pipeline
    python main.py --evaluate path/to/test.pdf  # Evaluate with a test document
"""
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Document Q&A Assistant")
    parser.add_argument("--api", action="store_true", help="Launch FastAPI server only")
    parser.add_argument("--both", action="store_true", help="Launch both API and UI")
    parser.add_argument("--evaluate", nargs="?", const=True, default=False,
                        help="Run evaluation pipeline (optionally with a test file path)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    # --- Evaluation Mode ---
    if args.evaluate:
        from evaluation import run_full_evaluation
        test_file = args.evaluate if isinstance(args.evaluate, str) else None
        run_full_evaluation(test_file)
        return

    # --- API Only Mode ---
    if args.api:
        import uvicorn
        from config import FASTAPI_HOST, FASTAPI_PORT
        print(f"Starting FastAPI server on http://{FASTAPI_HOST}:{FASTAPI_PORT}")
        print(f"API docs at http://{FASTAPI_HOST}:{FASTAPI_PORT}/docs")
        uvicorn.run("server:app", host=FASTAPI_HOST, port=FASTAPI_PORT, reload=False)
        return

    # --- Both API + UI Mode ---
    if args.both:
        import threading
        import uvicorn
        from config import FASTAPI_HOST, FASTAPI_PORT
        from ui import build_ui
        from vector_store import load_vectorstore

        load_vectorstore()

        # Start FastAPI in a background thread
        def run_api():
            uvicorn.run("server:app", host=FASTAPI_HOST, port=FASTAPI_PORT, reload=False)

        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        print(f"[Main] FastAPI started on http://{FASTAPI_HOST}:{FASTAPI_PORT}")

        # Start Gradio in the main thread
        demo = build_ui()
        demo.launch(share=args.share, server_port=7860)
        return

    # --- Default: Gradio UI Only ---
    from ui import launch_ui
    print("Starting Gradio UI...")
    launch_ui(share=args.share)


if __name__ == "__main__":
    main()
