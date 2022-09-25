import os
import numpy as np
import pandas as pd
from representation.label_factory import *



class LabelsCalculator:
    def __init__(self) -> None:
        pass
    
    def calculate_labels(self, df):
        # 对于涨跌幅 做放大1000倍数处理
        df = df.loc[:, ["DateTime", "LastPx", "MidPx", "Volume"]]

        # 1. 信号结束点出涨跌幅
        df.loc[:, "y_midpx_30s_at"] = px_change(df, tick_mode="MidPx", interval_seconds=30, tick_interval_seconds=3)
        df.loc[:, "y_midpx_3min_at"] = px_change(df, tick_mode="MidPx", interval_seconds=180, tick_interval_seconds=3)
        df.loc[:, "y_midpx_5min_at"] = px_change(df, tick_mode="MidPx", interval_seconds=300, tick_interval_seconds=3)

        # 2. 信号周期内，最大涨跌幅
        df.loc[:, "y_midpx_30s_interval_high"] = px_change_interval(df, direction_mode='high', tick_mode="MidPx", interval_seconds=30, tick_interval_seconds=3)
        df.loc[:, "y_midpx_30s_interval_low"] = px_change_interval(df, direction_mode='low', tick_mode="MidPx", interval_seconds=30, tick_interval_seconds=3)

        df.loc[:, "y_midpx_3min_interval_high"] = px_change_interval(df, direction_mode='high', tick_mode="MidPx", interval_seconds=180, tick_interval_seconds=3)
        df.loc[:, "y_midpx_3min_interval_low"] = px_change_interval(df, direction_mode='low', tick_mode="MidPx", interval_seconds=180, tick_interval_seconds=3)

        df.loc[:, "y_midpx_5min_interval_high"] = px_change_interval(df, direction_mode='high', tick_mode="MidPx", interval_seconds=300, tick_interval_seconds=3)
        df.loc[:, "y_midpx_5min_interval_low"] = px_change_interval(df, direction_mode='low', tick_mode="MidPx", interval_seconds=300, tick_interval_seconds=3)

        # 最后剔除非标签列
        df.drop(["LastPx", "MidPx", "Volume"], axis=1, inplace=True)
        return df
    