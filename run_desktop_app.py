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


def show_update_available(installed_version, target_version):
    print(
        f"[startup-update] updating from {installed_version} to {target_version}",
        flush=True,
    )


def check_for_startup_update(*, force_check=False):
    try:
        from desktop_app_source_updater import format_update_message, run_startup_update
        from startup_update_config import build_startup_update_config

        config = build_startup_update_config(
            base_path,
            on_update_available=show_update_available,
        )
        result = run_startup_update(config, force_check=force_check)
        update_message = format_update_message(result)
        if update_message:
            print(f"[startup-update] {update_message}", flush=True)
        return result
    except Exception as exc:
        print(f"[startup-update] skipped: {exc}", flush=True)
        return None


def run_dash():
    app.run(
        host="127.0.0.1",
        port=PORT,
        debug=False,
        dev_tools_hot_reload=False,
    )


def main():
    global app, WINDOW_CONFIG, PORT

    multiprocessing.freeze_support()
    explicit_update_check = "--check-update" in sys.argv[1:]
    update_result = check_for_startup_update(force_check=explicit_update_check)

    if explicit_update_check:
        if update_result is None:
            return 1
        print(f"[startup-update] status: {update_result.status}", flush=True)
        if update_result.status not in {"updated", "up-to-date"}:
            return 1
        from fp_analysis_app import VERSION

        print(f"update check ok: {VERSION}", flush=True)
        return 0

    from fp_analysis_app import VERSION
    from fp_analysis_app.app_dev import app
    from fp_analysis_app.config import WINDOW_CONFIG, PORT

    if "--smoke" in sys.argv[1:]:
        print(f"smoke ok: {VERSION}", flush=True)
        return 0

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
