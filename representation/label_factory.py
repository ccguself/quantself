import os
import numpy as np
import pandas as pd


def px_change(df, tick_mode="MidPx", interval_seconds=300, tick_interval_seconds=3):
    """
    结尾处涨跌幅
    """
    n_tick = interval_seconds // tick_interval_seconds
    return df[tick_mode].diff(n_tick) / df[tick_mode].shift(n_tick) * 1000


def px_change_interval(df, direction_mode="high", tick_mode="MidPx", interval_seconds=300, tick_interval_seconds=3):
    """
    区间内最大涨跌幅
    """
    def px_change_max_utils(x, mode):
        if mode == 'high':
            return (x.max() / x.iloc[0] - 1) * 1000
        elif mode == 'low':
            return (x.min() / x.iloc[0] - 1) * 1000
    n_tick = interval_seconds // tick_interval_seconds
    return df.loc[:, tick_mode].rolling(n_tick).apply(lambda x:px_change_max_utils(x, direction_mode))

