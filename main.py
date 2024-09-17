# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 23:44:51 2024

@author: yzhao
"""

from threading import Timer
from functools import partial

from app_src.app import app, open_browser


if __name__ == "__main__":
    PORT = 8050
    VERSION = "v0.11.0"
    Timer(1, partial(open_browser, PORT)).start()
    app.title = f"Sleep Scoring App {VERSION}"
    app.run_server(debug=False, port=PORT)
