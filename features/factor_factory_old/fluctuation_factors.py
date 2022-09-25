import numpy as np
import talib as ta
import pandas as pd
from .utils import ts_rank, scale

def short_term_amp(df, interval_seconds=30, tick_interval_seconds=3):
    """
    在 interval_seconds 内的短期振幅
    """
    n_tick = interval_seconds // tick_interval_seconds
    st_max = ta.MAX(df['LastPx'], n_tick)
    st_min = ta.MIN(df['LastPx'], n_tick)
    return (st_max - st_min) / (df['LastPx'].shift(n_tick)) * 1000


def short_term_volatility(df, interval_seconds=30, tick_interval_seconds=3):
    """
    在 interval_seconds 内的短期涨跌幅波动率
    """
    n_tick = interval_seconds // tick_interval_seconds
    return ta.VAR(df['LastPx'].diff(n_tick) / df['LastPx'].shift(n_tick) * 1000,
                  n_tick)


def stddev(df, interval_seconds=30, tick_interval_seconds=3):
    """
    STDDEV
    """
    n_tick = interval_seconds // tick_interval_seconds
    df['Stddev_{}'.format(interval_seconds)] = ta.STDDEV(
        df['LastPx'], timeperiod=n_tick, nbdev=1)
    return df


def high_volatility_feature_up(df, interval_seconds=30, tick_interval_seconds=3):
    """
    高频上行波动
    """
    n_tick = interval_seconds // tick_interval_seconds
    # 把return放大scale倍
    df['ret'] = np.log(df['LastPx']).diff(1) * scale
    df['indicator_pos'] = (df['ret'] > 0) * 1
    df['indicator_neg'] = (df['ret'] < 0) * 1
    df['high_volatility_up_{}'.format(interval_seconds)] = (ta.SUM(df['ret'] * df['indicator_pos'] ** 2, n_tick) + 1e-10) / (ta.SUM(
        df['ret'] ** 2, n_tick) + 1e-10)
    df.drop(columns=['ret', 'indicator_pos', 'indicator_neg'], inplace=True)
    return df


def high_volatility_feature_down(df, interval_seconds=30, tick_interval_seconds=3):
    """
    高频下行波动
    """
    n_tick = interval_seconds // tick_interval_seconds
    # 把return放大scale倍
    df['ret'] = np.log(df['LastPx']).diff(1) * scale
    df['indicator_pos'] = (df['ret'] > 0) * 1
    df['indicator_neg'] = (df['ret'] < 0) * 1
    df['high_volatility_down_{}'.format(interval_seconds)] = (ta.SUM(df['ret'] * df['indicator_neg'] ** 2, n_tick) + 1e-10) / (ta.SUM(
        df['ret'] ** 2, n_tick) + 1e-10)
    df.drop(columns=['ret', 'indicator_pos', 'indicator_neg'], inplace=True)
    return df


def high_volatility_feature_up_ratio(df, interval_seconds=30, tick_interval_seconds=3):
    """
    高频上行波动占比
    """
    n_tick = interval_seconds // tick_interval_seconds
    # 把return放大scale倍
    df['ret'] = np.log(df['LastPx']).diff(1) * scale
    df['indicator_pos'] = (df['ret'] > 0) * 1
    df['indicator_neg'] = (df['ret'] < 0) * 1
    df['high_volatility_up_ratio_{}'.format(interval_seconds)] = (ta.SUM(df['ret'] * df['indicator_pos'] ** 2, n_tick) + 1e-10) / (ta.SUM(
        df['ret'] ** 2, n_tick) + 1e-10)
    df.drop(columns=['ret', 'indicator_pos', 'indicator_neg'], inplace=True)
    return df


def high_volatility_feature_down_ratio(df, interval_seconds=30, tick_interval_seconds=3):
    """
    高频下行波动占比
    """
    n_tick = interval_seconds // tick_interval_seconds
    # 把return放大scale倍
    df['ret'] = np.log(df['LastPx']).diff(1) * scale
    df['indicator_pos'] = (df['ret'] > 0) * 1
    df['indicator_neg'] = (df['ret'] < 0) * 1
    df['high_volatility_down_ratio_{}'.format(interval_seconds)] = (ta.SUM(df['ret'] * df['indicator_neg'] ** 2, n_tick) + 1e-10) / (ta.SUM(
        df['ret'] ** 2, n_tick) + 1e-10)
    df.drop(columns=['ret', 'indicator_pos', 'indicator_neg'], inplace=True)
    return df