import os
import base64
from dotenv import load_dotenv
from json import dumps, load
import json
import math
import time
from pprint import pprint
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import requests # supports CoinMarketCap
# coinbase api
from coinbase.rest import RESTClient
# from mailjet_rest import Client
# parse CLI args
import argparse
# related to price change % logic
import glob

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils.price_helpers import calculate_trading_range_percentage

def plot_graph(
    enabled,
    current_timestamp,
    symbol,
    price_data,
    min_price,
    max_price,
    trade_range_percentage,
    entry_price
):
    if not enabled:
        if plt.get_fignums():
            plt.close('all')
        return

    plt.figure(figsize=(9, 7))

    if entry_price > 0:
        plt.axhline(y=entry_price, color='m', linewidth=1.2, linestyle='-', label=f"entry price ({entry_price})")

    plt.plot(list(price_data), marker=',', label='price', c='black')

    colors = ['cadetblue', 'blue', 'green', 'orange', 'lime', 'lavender']

    #
    #
    price_range = max_price - min_price
    buffer = price_range * 0.05

    plt.gca().set_ylim(min_price - buffer, max_price + buffer)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

    plt.title(f"{symbol}")
    plt.xlabel("time range")
    plt.ylabel(f"price range: {trade_range_percentage}%")
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True)

    filename = os.path.join("./screenshots", f"{symbol}_chart_{current_timestamp}.png")
    if os.path.exists(filename):
        current_time = time.time()
        twenty_minutes_ago = current_time - 1200  # 20 minutes in seconds
        file_mod_time = os.path.getmtime(filename)
        if file_mod_time > twenty_minutes_ago:
            print(f"Chart for {symbol} is less than 20 minutes old, not saving a new one.")
            return
        os.remove(filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"Chart saved as {filename}")
