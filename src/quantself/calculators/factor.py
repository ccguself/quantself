import os
import numpy as np
import pandas as pd
from utils import *


def px_change_ratio(df, interval_seconds=30, tick_interval_seconds=3):
    """
    在 interval_seconds 内的涨跌幅
    """
    n_tick = interval_seconds // tick_interval_seconds
    return df['LastPx'].diff(n_tick) / df['LastPx'].shift(n_tick)
