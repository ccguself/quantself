import numpy as np
import talib as ta
import pandas as pd
from .utils import *
from pyfinance.ols import PandasRollingOLS

def gtja_alpha002(df, interval_seconds=30, tick_interval_seconds=3):
    """
    国泰君安191因子
    """
    n_tick = interval_seconds // tick_interval_seconds

    df['gtja_alpha002_{}'.format(interval_seconds)] = -1 * (((df['LastPx']-df['LowPx']) - (df['HighPx']-df['LowPx'])+ 1e-10)/(df['HighPx']-df['LowPx']+ 1e-10)).shift(1)
    return df


def gtja_alpha013(df, interval_seconds=30, tick_interval_seconds=3):
    """
    国泰君安191因子
    """
    n_tick = interval_seconds // tick_interval_seconds
    df['gtja_alpha013_{}'.format(interval_seconds)] = (df['HighPx'] * df['LowPx'])**0.5 - (df['WeightedAvgBuyPx']+df['WeightedAvgSellPx']) * 0.5
    return df


def gtja_alpha014(df, interval_seconds=30, tick_interval_seconds=3):
    """
    国泰君安191因子
    """
    n_tick = interval_seconds // tick_interval_seconds
    df['gtja_alpha014_{}'.format(interval_seconds)] = df['LastPx'] - df['LastPx'].shift(n_tick)
    return df


def gtja_alpha015(df, interval_seconds=30, tick_interval_seconds=3):
    """
    国泰君安191因子
    """
    n_tick = interval_seconds // tick_interval_seconds

    df['gtja_alpha015_{}'.format(interval_seconds)] = (df['OpenPx']+ 1e-10) / (df['LastPx'].shift(n_tick)+ 1e-10) - 1
    return df


#
def gtja_alpha046(df, interval_seconds=30, tick_interval_seconds=3):
    '''
    国泰君安191因子
    '''

    n_tick = interval_seconds // tick_interval_seconds
    df['gtja_alpha046_{}'.format(interval_seconds)] = (ta.SMA(df['LastPx'], 3)+ ta.SMA(df['LastPx'], 6) + ta.SMA(df['LastPx'], 9) + ta.SMA(df['LastPx'], 12)) * 0.25 / df['LastPx']
    return df


def gtja_alpha047(df, interval_seconds=30, tick_interval_seconds=3):
    """
    国泰君安191因子
    """
    n_tick = interval_seconds // tick_interval_seconds

    df['gtja_alpha047_{}'.format(interval_seconds)] = ta.SMA(100 * (ta.MAX(df['HighPx'], 6) - df['LastPx']+ 1e-10) / (ta.MAX(df['HighPx'], 6) - ta.MIN(df['LowPx'], 6)+ 1e-10), n_tick)
    return df


def gtja_alpha057(df, interval_seconds=30, tick_interval_seconds=3):
    """
    国泰君安191因子
    """
    n_tick = interval_seconds // tick_interval_seconds
    df['gtja_alpha057_{}'.format(interval_seconds)] = ta.SMA(100 * (df['LastPx']-ta.MIN(df['LowPx'], 9)+ 1e-10) / (ta.MAX(df['HighPx'],9)-ta.MIN(df['LowPx'], 9)+ 1e-10), n_tick)
    return df


def gtja_alpha070(df, interval_seconds=30, tick_interval_seconds=3):
    """
    国泰君安191因子
    """
    n_tick = interval_seconds // tick_interval_seconds

    df['gtja_alpha070_{}'.format(interval_seconds)] = ta.STDDEV(df['TotalValueTrade'], 6)
    return df

def gtja_alpha078(df, interval_seconds=30, tick_interval_seconds=3):
    """
    国泰君安191因子
    """
    n_tick = interval_seconds // tick_interval_seconds
    df['gtja_alpha078_{}'.format(interval_seconds)] = ((df['HighPx']+df['LowPx']+df['LastPx']) - ta.SMA((df['HighPx']+df['LowPx']+df['LastPx']), 3)
    + 1e-10) / (ta.SMA(abs(df['LastPx']-ta.SMA((df['HighPx']+df['LowPx']+df['LastPx']), 3)),12) * 0.015+ 1e-10)
    return df


def gtja_alpha171(df, interval_seconds=30, tick_interval_seconds=3):
    """
    国泰君安191因子
    """
    n_tick = interval_seconds // tick_interval_seconds

    df['gtja_alpha171_{}'.format(interval_seconds)] = -1 * ((df['LowPx'] - df['LastPx']) * (df['OpenPx']**5)+ 1e-10) / ((
        df['LastPx'] - df['HighPx']) * (df['LastPx']**5)+ 1e-10)
    return df

# 东方证券遗传算法因子
def get_df_alpha10_13(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = np.log(df['TotalVolumeTrade'].diff(1)) * scale
    # df['alpha_10_13_{}'.format(n_tick)] = ta.STDDEV(np.sqrt(volume), n_tick)
    return ta.STDDEV(np.sqrt(volume), n_tick).values


# 东方证券遗传算法因子
def get_df_alpha4_97(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds

    volume = np.log(df['TotalVolumeTrade'].diff(1)) * scale
    ret = np.log(df['LastPx'].diff(1)) * scale
    # df['alpha_4_97_{}'.format(n_tick)] = np.sqrt(volume) - ta.SUM(ret, n_tick)
    return (np.sqrt(volume) - ta.SUM(ret, n_tick)).values


# 东方证券遗传算法因子
def get_df_alpha10_11(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds

    volume = np.log(df['TotalVolumeTrade'].diff(1)) * scale
    # df['alpha_10_11'] = np.sqrt(volume)
    return np.sqrt(volume).values


# 东方证券遗传算法因子
def get_df_alpha1_92(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds

    abs_volume = abs(df['TotalVolumeTrade'].diff(1)) * scale
    # df['alpha_1_92'] = abs(ta.SUM(np.log(abs_volume), n_tick))
    return abs(ta.SUM(np.log(abs_volume), n_tick)).values


# 东方证券遗传算法因子
def get_df_alpha9_9(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds

    volume = np.log(df['TotalVolumeTrade'].diff(1)) * scale

    # df['alpha_9_9'] = np.sqrt(ta.STDDEV(ta.STDDEV(volume, n_tick*6), n_tick))
    return np.sqrt(ta.STDDEV(ta.STDDEV(volume, n_tick * 6), n_tick)).values


# 东方证券遗传算法因子
def get_df_alpha3_67(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds

    volume = np.log(df['TotalVolumeTrade'].diff(1)) * scale

    # df['alpha_3_67_{}'.format(n_tick)] = volume + ta.MIN(df['HighPx'], df['LastPx'], n_tick)
    return (volume + ta.MIN(df['HighPx'], df['LastPx'], n_tick)).values


# 东方证券遗传算法因子
def get_df_alpha4_43(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds

    volume = np.log(df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(n_tick * 4)) * scale

    # df['alpha_4_43_{}'.format(n_tick)] = ta.MAX(volume, n_tick)
    return ta.MAX(volume, n_tick).values


# 东方证券遗传算法因子
def get_df_alpha4_34(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds

    volume = np.log(df['TotalVolumeTrade'].diff(1)) * scale

    # df['alpha_4_34_{}'.format(n_tick)] = ta.SUM(ta.CORREL(volume, df['HighPx'], n_tick*4), n_tick)
    return ta.SUM(ta.CORREL(volume, df['HighPx'], n_tick * 4), n_tick).values


# 东方证券遗传算法因子
def get_df_alpha7_80(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds

    ret = np.log(df['LastPx'].diff(1)) * scale

    # df['alpha_7_80_{}'.format(n_tick)] = ta.MAX(ret, n_tick)
    return ta.MAX(ret, n_tick).values


#####################
# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha1(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    vwap = ta.SMA(df['LastPx'], n_tick)
    amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    temp1 = ta.SMA(amount, n_tick * 2)
    temp2 = ta.MAX(ta.STDDEV(volume * vwap, n_tick * 2), n_tick)

    df['tf_alpha1_{}'.format(interval_seconds)] = ta.MAX(ts_cov(temp1, temp2, n_tick * 2), n_tick * 2)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha2(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    temp1 = ta.CORREL((df['HighPx'] + volume), df['HighPx'], n_tick)
    temp2 = ta.CORREL(df['LastPx'] - df['LastPx'].shift(2), amount - amount.shift(2), n_tick)
    temp3 = ta.CORREL(df['LastPx'], temp2)

    df['tf_alpha2_{}'.format(interval_seconds)] = temp3 - temp1
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha3(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    vwap = ta.SMA(df['LastPx'], n_tick)
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    temp1 = vwap - vwap.shift(5)
    temp2 = np.sqrt(ts_rank(-1 * df['LastPx'], n_tick))

    df['tf_alpha3_{}'.format(interval_seconds)] = temp1 - temp2
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha4(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    temp1 = ta.SMA(amount, n_tick * 2)
    temp2 = ta.MAX(df['LowPx'], n_tick)
    df['tf_alpha4_{}'.format(interval_seconds)] = ta.MAX(ts_cov(temp2, temp2, n_tick * 2), n_tick * 2)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha5(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    temp1 = ts_cov(volume, amount, n_tick)
    temp2 = ta.SMA(PandasRollingOLS(amount, df['HighPx'], n_tick * 2).beta, n_tick)
    df['tf_alpha5_{}'.format(interval_seconds)] = (temp1 + 1e-10) / (temp2 + 1e-10)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha6(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    temp1 = ta.CORREL(amount, df['HighPx'], n_tick * 2)

    df['tf_alpha6_{}'.format(interval_seconds)] = ta.MAX(temp1, n_tick)
    return df

# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha7(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    temp1 = ta.CORREL(amount, df['HighPx'], n_tick * 2)

    df['tf_alpha7_{}'.format(interval_seconds)] = ta.MAX(temp1, n_tick)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha8(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    temp1 = ta.CORREL(amount, df['HighPx'], n_tick * 2) * np.arctan(volume + 1e-10)
    temp2 = df['LastPx'] - df['LastPx'].shift(n_tick)
    df['tf_alpha8_{}'.format(interval_seconds)] = np.sqrt(np.minimum(np.log(temp1), temp2))
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha8(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    temp1 = ta.CORREL(amount, df['HighPx'], n_tick * 2) * np.arctan(volume + 1e-10)
    temp2 = df['LastPx'] - df['LastPx'].shift(n_tick)

    df['tf_alpha8_{}'.format(interval_seconds)] = np.sqrt(np.minimum(np.log(temp1), temp2))
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha9(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    # amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    temp1 = np.maximum(np.sign(volume), volume - volume.shift(n_tick))
    temp2 = df['HighPx'] - df['HighPx'].shift(n_tick)

    df['tf_alpha9_{}'.format(interval_seconds)] = np.minimum(temp1, temp2)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha10(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    # amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    temp1 = ta.STDDEV(df['LastPx'], n_tick)
    temp2 = np.maximum(temp1, ta.MAX(volume, n_tick))

    df['tf_alpha10_{}'.format(interval_seconds)] = ta.STDDEV(temp1 + temp2, n_tick)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha11(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds

    df['tf_alpha11_{}'.format(interval_seconds)] = np.log(df['LastPx']) - np.log(df['LastPx']).shift(n_tick)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha12(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds

    temp1 = df['LowPx'] - df['LowPx'].shift(n_tick)
    temp2 = df['LastPx'] - df['LastPx'].shift(n_tick)

    df['tf_alpha12_{}'.format(interval_seconds)] = np.minimum(temp1, temp2)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha14(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)

    df['tf_alpha14_{}'.format(interval_seconds)] = ta.CORREL(np.log(volume), df['LastPx'])
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha21(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    vwap = ta.SMA(df['LastPx'], n_tick)
    temp1 = np.minimum(vwap, ta.STDDEV(volume, n_tick))
    temp2 = ta.MAX(df['LastPx'], n_tick)

    df['tf_alpha21_{}'.format(interval_seconds)] = (temp1 + 1e-10) / (temp2 + 1e-10)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha23(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    temp1 = df['LastPx'] - df['LastPx'].shift(n_tick)
    temp11 = temp1 - temp1.shift(n_tick // 2)
    temp2 = ta.MAX(amount, n_tick)

    df['tf_alpha23_{}'.format(interval_seconds)] = temp11 + temp2
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha26(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    # amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    vwap = ta.SMA(df['LastPx'], n_tick)

    df['tf_alpha26_{}'.format(interval_seconds)] = ta.CORREL(volume, vwap, n_tick)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha33(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    # amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    vwap = ta.SMA(df['LastPx'], n_tick)

    df['tf_alpha33_{}'.format(interval_seconds)] = ts_cov(vwap, volume, n_tick)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha36(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    # amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    vwap = ta.SMA(df['LastPx'], n_tick)

    df['tf_alpha36_{}'.format(interval_seconds)] = ta.CORREL(ta.SMA(volume, n_tick), vwap, n_tick * 2)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha39(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    vwap = ta.SMA(df['LastPx'], n_tick)

    df['tf_alpha39_{}'.format(interval_seconds)] = ta.SMA(ts_cov(vwap, amount, n_tick), n_tick * 2) + volume
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha42(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    # amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    # vwap = ta.SMA(df['LastPx'], n_tick)

    df['tf_alpha42_{}'.format(interval_seconds)] = ta.CORREL(df['HighPx'] - df['HighPx'].shift(n_tick), -1 * volume,
                                                             n_tick)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha47(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    pct_close = df['LastPx'].diff(1)
    pct_high = df['HighPx'].diff(1)

    df['tf_alpha47_{}'.format(interval_seconds)] = np.minimum(pct_close, pct_high)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha50(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    # volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    # amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    vwap = ta.SMA(df['LastPx'], n_tick)

    df['tf_alpha50_{}'.format(interval_seconds)] = ta.MAX(vwap, n_tick) / df['HighPx']
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha56(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    # amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    # vwap = np.log(ta.SMA(df['LastPx'], n_tick))
    temp1 = volume - volume.shift(n_tick)
    temp2 = ta.CORREL(volume, df['HighPx'], n_tick) + df['LastPx']

    df['tf_alpha56_{}'.format(interval_seconds)] = ta.CORREL(temp1, temp2, n_tick)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha57(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    # amount = df['TotalValueTrade'] - df['TotalValueTrade'].shift(1)
    vwap = np.log(ta.SMA(df['LastPx'], n_tick))
    temp1 = df['LowPx'] + np.maximum(volume, df['LowPx'])
    temp2 = (vwap - vwap.shift(n_tick) + 1e-10) / (vwap + 1e-10)

    df['tf_alpha57_{}'.format(interval_seconds)] = ta.CORREL(temp1, temp2, n_tick)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha70(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = np.log(df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1) + 1e-10)

    df['tf_alpha70_{}'.format(interval_seconds)] = ta.STDDEV(volume - volume.shift(n_tick * 2), n_tick)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha71(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = -1 * (df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1) + 1e-10)

    df['tf_alpha71_{}'.format(interval_seconds)] = ta.STDDEV(volume - volume.shift(n_tick * 2), n_tick)
    return df


# 天风证券遗传算法因子：市场微观结构探析：分时K线中的alpha
def tf_alpha87(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    vwap = np.log(ta.SMA(df['LastPx'], n_tick))

    df['tf_alpha87_{}'.format(interval_seconds)] = (vwap - vwap.shift(n_tick) + 1e-10) / (vwap + 1e-10)
    return df

def xc_gep_factor_1(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds

    def ts_diff(x1, d=1):
        x1 = pd.Series(x1.squeeze())
        df = x1.diff(d).fillna(0.0)
        return df.squeeze()

    def ewm_mean(x1, d=10):
        x1 = pd.Series(x1.squeeze())
        return x1.ewm(d).mean()

    def gp_sig(x):
        return 1. / (1. + np.exp(-x))

    TotalSellNumber = df.apply(lambda x: np.sum(x['SellNumOrdersQueue']), axis=1)

    df['xc_gep_factor_1_{}'.format(interval_seconds)] = gp_sig(ewm_mean(df['LowPx'] * ts_diff(TotalSellNumber), n_tick))

    return df


def xc_gep_factor_2(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds

    def ts_diff(x1, d=1):
        x1 = pd.Series(x1.squeeze())
        df = x1.diff(d).fillna(0.0)
        return df.squeeze()

    def ewm_mean(x1, d=10):
        x1 = pd.Series(x1.squeeze())
        return x1.ewm(d).mean()

    TotalSellNumber = df.apply(lambda x: np.sum(x['SellNumOrdersQueue']), axis=1)

    df['xc_gep_factor_2_{}'.format(interval_seconds)] = ewm_mean(
        abs(df['WeightedAvgSellPx']) * ts_diff(TotalSellNumber), n_tick)

    return df


def xc_gep_factor_3(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds

    def ts_diff(x1, d=1):
        x1 = pd.Series(x1.squeeze())
        df = x1.diff(d).fillna(0.0)
        return df.squeeze()

    def ewm_mean(x1, d=10):
        x1 = pd.Series(x1.squeeze())
        return x1.ewm(d).mean()

    TotalSellNumber = df.apply(lambda x: np.sum(x['SellNumOrdersQueue']), axis=1)

    df['xc_gep_factor_3_{}'.format(interval_seconds)] = ts_diff(abs(ewm_mean(df['WeightedAvgSellPx'], n_tick)))

    return df

def alpha041(df):
    vwap = df['TotalValueTrade'] / df['TotalVolumeTrade']
    alpha = pow((df['HighPx'] * df['LowPx']), 0.5) - vwap
    df['alpha041'] = alpha * scale
    return df

def alpha054(df):
    inner = (df['LowPx'] - df['HighPx']).replace(0, -0.0001)
    alpha = -1 * (df['LowPx'] - df['LastPx']) * (df['OpenPx'] ** 5) / (inner * (df['LastPx'] ** 5))
    df['alpha054'] = alpha * scale
    return df

def alpha101(df):
    df['alpha101'] = (df['LastPx'] - df['OpenPx'] + 1e-10) / ((df['HighPx'] - df['LowPx']) + 1e-10) \
                     * scale
    return df

def alpha006(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'].diff(1)
    df['alpha006_{}'.format(interval_seconds)] = -1 * correlation(df['LastPx'], volume, n_tick)
    return df

def alpha007(df, interval_seconds=60, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'].diff(1)
    adv20 = sma(volume, n_tick)
    alpha = -1 * ts_rank(abs(delta(df['LastPx'], n_tick)), n_tick) * np.sign(delta(df['LastPx'], n_tick))
    alpha[(adv20 >= volume) & (alpha.notna())] = -1
    df['alpha007_{}'.format(interval_seconds)] = alpha
    return df

    #
def alpha009(df, interval_seconds=60, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    delta_close = delta(df['LastPx'], 1) * scale
    cond_1 = ts_min(delta_close, n_tick) > 0
    cond_2 = ts_max(delta_close, n_tick) < 0
    alpha = -1 * delta_close
    alpha[cond_1 | cond_2] = delta_close
    df['alpha009_{}'.format(interval_seconds)] = alpha
    return df


def alpha012(df, interval_seconds=3, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'].diff(1)
    # alpha = sign(delta(volume, n_tick)) * (-1 * delta(df['LastPx'], n_tick))
    alpha = np.sign(delta(volume, n_tick)) * (-1 * delta(df['LastPx'], n_tick)) * scale
    df['alpha012_{}'.format(interval_seconds)] = alpha
    return df

def alpha024(df, interval_seconds=300, tick_interval_seconds=3, threshold=0.001):
    n_tick = interval_seconds // tick_interval_seconds
    cond = ((delta(sma(df['LastPx'], n_tick), n_tick) + 1e-10) / (delay(df['LastPx'], n_tick) + 1e-10)) \
                                        <= threshold # 涨跌幅
    alpha = -1 * delta(df['LastPx'], n_tick)
    alpha[cond] = -1 * (df['LastPx'] - ts_min(df['LastPx'], n_tick))
    df['alpha024_{}'.format(interval_seconds)] = alpha
    return df

def alpha032(df, interval_seconds=60, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    vwap = df['TotalValueTrade'] / df['TotalVolumeTrade']
    alpha = ((sma(df['LastPx'], n_tick) / n_tick) - df['LastPx']) + (
                n_tick * correlation(vwap, delay(df['LastPx'], n_tick), n_tick)) # correlation
    df['alpha032_{}'.format(interval_seconds)] = alpha
    return df

def alpha035(df, interval_seconds=60, tick_interval_seconds=3):
    volume = df['TotalVolumeTrade'].diff(1)
    n_tick = interval_seconds // tick_interval_seconds
    returns = df['LastPx'].diff(n_tick) / df['LastPx'].shift(n_tick) * scale
    alpha = ((ts_rank(volume, n_tick) *
             (1 - ts_rank(df['LastPx'] + df['HighPx'] - df['LowPx'], n_tick // 2))) *
            (1 - ts_rank(returns, n_tick)))
    df['alpha035_{}'.format(interval_seconds)] = alpha
    return df

def alpha049(df, interval_seconds=60, tick_interval_seconds=3, threshold = 1):
    n_tick = interval_seconds // tick_interval_seconds
    n_tick = n_tick // 2
    inner = (((delay(df['LastPx'], n_tick * 2) - delay(df['LastPx'], n_tick)) / n_tick) - ((delay(df['LastPx'], n_tick) - df['LastPx']) / n_tick))
    inner *= scale
    alpha = (-1 * delta(df['LastPx'], n_tick)).copy()
    alpha[(inner < (-1 * threshold)) & (alpha.notna())] = 1
    df['alpha049_{}'.format(interval_seconds)] = alpha
    return df


def alpha051(df, interval_seconds=60, tick_interval_seconds=3, threshold = 0.5):
    n_tick = interval_seconds // tick_interval_seconds
    n_tick = n_tick // 2
    inner = (((delay(df['LastPx'], n_tick * 2) - delay(df['LastPx'], n_tick)) / n_tick) - ((delay(df['LastPx'], n_tick) - df['LastPx']) / n_tick))
    inner *= scale
    alpha = (-1 * delta(df['LastPx'], n_tick)).copy()
    alpha[(inner < (-1 * threshold)) & (alpha.notna())] = 1
    df['alpha051_{}'.format(interval_seconds)] = alpha
    return df


def alpha084(df, interval_seconds=60, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    vwap = df['TotalValueTrade'] / df['TotalVolumeTrade']
    cond1 = ts_rank((vwap - ts_max(vwap, n_tick // 2)), n_tick)
    cond2 = delta(df['LastPx'], n_tick // 4)
    alpha = pow(cond1, cond2)
    df['alpha084_{}'.format(interval_seconds)] = alpha
    df.ix[(cond1.isna()) | (cond2.isna()), 'alpha084_{}'.format(interval_seconds)] = np.nan
    return df


## ----是否能修改参数？？ -----------------####################
def alpha021(df, interval_seconds=3, tick_interval_seconds=3):
    volume = df['TotalVolumeTrade'].diff(1)
    cond_1 = sma(df['LastPx'], 8) + stddev(df['LastPx'], 8) < sma(df['LastPx'], 2)
    cond_2 = (sma(volume, 20) / volume) < 1
    df['alpha021_{}'.format(interval_seconds)] = np.ones_like(df['LastPx'])
    df.ix[cond_1 | cond_2, 'alpha021_{}'.format(interval_seconds)] = -1
    return df

def alpha023(df, interval_seconds=60, tick_interval_seconds=3):
    n_tick = interval_seconds / tick_interval_seconds
    cond = (sma(df['HighPx'], n_tick)) < df['HighPx']
    df['alpha023_{}'.format(interval_seconds)] = np.zeros_like(df['LastPx'])
    df.ix[cond, 'alpha023_{}'.format(interval_seconds)] = -1 * delta(df['HighPx'], n_tick)
    return df

def alpha026(df, interval_seconds=30, tick_interval_seconds=3):
    volume = df['TotalVolumeTrade'].diff(1)
    n_tick = interval_seconds // tick_interval_seconds
    alpha = correlation(ts_rank(volume, n_tick), ts_rank(df['HighPx'], n_tick), n_tick)
    df['alpha026_{}'.format(interval_seconds)] = -1 * ts_max(alpha, n_tick)
    return df

def alpha028(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'].diff(1)
    adv20 = sma(volume, n_tick)
    temp = correlation(adv20, df['LowPx'], n_tick // 2)
    alpha = (adv20 + ((df['HighPx'] + df['LowPx']) / 2)) - df['LastPx']
    df['alpha028_{}'.format(interval_seconds)] = alpha
    return df

def alpha043(df, interval_seconds=60, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    volume = df['TotalVolumeTrade'].diff(1)
    adv20 = sma(volume, n_tick)
    alpha = ts_rank((volume + 1e-10) / (adv20 + 1e-10), n_tick) * ts_rank((-1 * delta(df['LastPx'], n_tick)), n_tick)
    df['alpha043_{}'.format(interval_seconds)] = alpha
    return df

def alpha046(df, interval_seconds=60, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    n_tick = n_tick // 2
    inner = ((delay(df['LastPx'], n_tick * 2) - delay(df['LastPx'], n_tick)) / n_tick) - ((delay(df['LastPx'], n_tick) - df['LastPx']) / n_tick)
    inner *= 1000
    # alpha = (-1 * delta(df['LastPx']))
    # alpha[inner < 0] = 1
    # alpha[inner > 0.25] = -1
    df['alpha046_{}'.format(interval_seconds)] = inner
    return df


def alpha053(df, interval_seconds=60, tick_interval_seconds=3):
    inner = (df['LastPx'] - df['LowPx']).replace(0, 0.0001)
    alpha = -1 * delta((((df['LastPx'] - df['LowPx']) - (df['HighPx'] - df['LastPx'])) / inner), 9)
    df['alpha053_{}'.format(interval_seconds)] = alpha
    return df
