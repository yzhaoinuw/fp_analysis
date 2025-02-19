# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 23:44:51 2024

@author: yzhao
"""

import os
import sys
from threading import Timer
from functools import partial


# Determine the base path depending on whether the script is frozen
if getattr(sys, "frozen", False):
    # The application is frozen
    base_path = os.path.dirname(sys.executable)
else:
    # The application is not frozen
    base_path = os.path.abspath(os.path.dirname(__file__))

# Add the base path to sys.path to find app_src
sys.path.insert(0, base_path)

if __name__ == "__main__":
    try:
        from app_src.app import app, open_browser

        PORT = 8050
        Timer(1, partial(open_browser, PORT)).start()
        app.run_server(
            debug=True, port=PORT, use_reloader=False, dev_tools_hot_reload=False
        )

    except ImportError as e:
        print(f"Error importing app_src: {e}")
        # sys.exit(1)
