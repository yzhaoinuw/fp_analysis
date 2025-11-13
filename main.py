# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 12:54:31 2025

@author: yzhao
"""

import os
import sys
import multiprocessing
from threading import Timer
from functools import partial

# Prevent PyInstaller + diskcache double launch
multiprocessing.freeze_support()

# Determine base path
if getattr(sys, "frozen", False):
    base_path = os.path.dirname(sys.executable)
else:
    base_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, base_path)

if __name__ == "__main__":
    try:
        from fp_analysis_app.app import app, open_browser

        PORT = 8050

        # Open browser only in the main process (not diskcache workers)
        if multiprocessing.current_process().name == "MainProcess":
            Timer(1, partial(open_browser, PORT)).start()

        # No reloader, no hot reload
        app.run(debug=False, port=PORT, use_reloader=False, dev_tools_hot_reload=False)

    except ImportError as e:
        print(f"Error importing fp_analysis_app: {e}")
