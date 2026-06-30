# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 13:31:54 2025

@author: yzhao
"""

import os
import sys
import threading
import multiprocessing

import webview


if getattr(sys, "frozen", False):
    # Running as packaged .exe → base path is folder containing executable
    base_path = os.path.dirname(sys.executable)
else:
    # Running as normal script → base path is folder containing this file
    base_path = os.path.abspath(os.path.dirname(__file__))

# Insert base_path FIRST so that fp_analysis_app/ next to .exe overrides bundled version
sys.path.insert(0, base_path)

try:
    from startup_update import format_startup_update_message, run_startup_update

    update_result = run_startup_update(base_path)
    update_message = format_startup_update_message(update_result)
    if update_message:
        print(f"[startup-update] {update_message}", flush=True)
except Exception as exc:
    print(f"[startup-update] skipped: {exc}", flush=True)


def run_dash():
    app.run(
        host="127.0.0.1",
        port=PORT,
        debug=False,
        dev_tools_hot_reload=False,
    )


if __name__ == "__main__":
    from fp_analysis_app import VERSION
    from fp_analysis_app.app_dev import app
    from fp_analysis_app.config import WINDOW_CONFIG, PORT

    multiprocessing.freeze_support()
    t = threading.Thread(target=run_dash, daemon=True)
    t.start()

    # This is the window `webview.windows[0]` will refer to
    webview.create_window(
        f"FP Analysis App {VERSION}",
        f"http://127.0.0.1:{PORT}",
        **WINDOW_CONFIG,
    )

    # Start pywebview (Windows → force edgechromium, others → auto)
    if sys.platform == "win32":
        webview.start(gui="edgechromium")
    else:
        webview.start()  # macOS/Linux auto-selects native renderer
