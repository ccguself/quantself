from functools import partial, update_wrapper
import talib as ta
import numpy as np
import time
import pandas as pd
scale = 1000







# 计算时间函数
def GetRunTime(func):
    def call_func(*args, **kwargs):
        begin_time = time.time()
        ret = func(*args, **kwargs)
        end_time = time.time()
        eclipse_time = end_time - begin_time
        print("{:<30}函数运行时间:{:.2f}s".format(str(func.__name__), eclipse_time))
        return ret
    return call_func


def relt(ds_a, ds_b):
    return (ds_a / ds_b - 1.0) * 100


def diff(ds, seconds, tick_interval=3):
    return ds.diff(seconds // tick_interval) / ds * 100


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


diff30 = wrapped_partial(diff, seconds=30)
diff60 = wrapped_partial(diff, seconds=60)
diff180 = wrapped_partial(diff, seconds=180)
diff300 = wrapped_partial(diff, seconds=300)





from scipy.stats import rankdata

def rolling_rank(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    :param na: numpy array.
    :return: The rank of the last value in the array.
    """
    return rankdata(na)[-1]

def ts_cov(df1, df2, window=10):
    """
    :param df1:
    :param df2:
    :param window:
    :return:
    """
    temp1 = df1.rolling(window=window, min_periods=1).mean()
    temp2 = df2.rolling(window=window, min_periods=1).mean()
    temp3 = (df1*df2).rolling(window=window, min_periods=1).mean()
    return temp3-temp1*temp2

def ts_rank(df, window=10):
    """
    Wrapper function to estimate rolling rank.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series rank over the past window days.
    """
    return df.rolling(window).apply(rolling_rank)

def delta(df, period=1):
    """
    Wrapper function to estimate difference.
    按参数求一列时间序列数据差值，period=1，今日减去昨日，以此类推
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)


def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    时间序列数据中第N天前的值
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)

def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    滑动窗口数据求和
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """

    return df.rolling(window).sum()


def sma(df, window=10):
    """
    Wrapper function to estimate SMA.
    滑动窗口求简单平均数
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return ta.SMA(df, timeperiod=window)


# def stddev(df, window=10):
#     """
#     Wrapper function to estimate rolling standard deviation.
#     滑动窗口求标准差
#     :param df: a pandas DataFrame.
#     :param window: the rolling window.
#     :return: a pandas DataFrame with the time-series min over the past 'window' days.
#     """
#     return df.rolling(window).std()


def correlation(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    滑动窗口求相关系数
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return ta.CORREL(x, y, timeperiod=window)


def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    滑动窗口求协方差
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)


def rolling_prod(na):
    """
    Auxiliary function to be used in pd.rolling_apply
    只在product函数中使用
    :param na: numpy array.
    :return: The product of the values in the array.
    """
    return np.prod(na)


def product(df, window=10):
    """
    Wrapper function to estimate rolling product.
    滑动窗口中的数据乘积
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series product over the past 'window' days.
    """
    return df.rolling(window).apply(rolling_prod)


def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    滑动窗口中的数据最小值
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return ta.MIN(df)


def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    滑动窗口中的数据最大值
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()


def rank(df):
    """
    Cross sectional rank
    排序，返回排序百分比数
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    return df.rank(axis=1, pct=True)
    # return df.rank(pct=True)


def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    滑动窗口中的数据最大值位置
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1


def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    滑动窗口中的数据最小值位置
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1


def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    df中从远及近分别乘以权重d，d-1，d-2，...，权重和为1
    例如：period=10时的权重列表
    [ 0.01818182,  0.03636364,  0.05454545,  0.07272727,  0.09090909,
        0.10909091,  0.12727273,  0.14545455,  0.16363636,  0.18181818]
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period,:]  # 本行有修订
    na_series = df.as_matrix()

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=['CLOSE'])  # 本行有修订
    # return pd.DataFrame(na_lwma, index=df.index, columns=df.keys())  # 本行有修订


def decay_linear_pn(df, period=10):
    """
    Linear weighted moving average implementation.
    df中从远及近分别乘以权重d，d-1，d-2，...，权重和为1
    例如：period=10时的权重列表
    [ 0.01818182,  0.03636364,  0.05454545,  0.07272727,  0.09090909,
        0.10909091,  0.12727273,  0.14545455,  0.16363636,  0.18181818]
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    print(np.shape(df))
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period,:]  # 本行有修订
    na_series = df.as_matrix()

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
        # return pd.DataFrame(na_lwma, index=df.index, columns=['CLOSE'])  # 本行有修订
    return pd.DataFrame(na_lwma, index=df.index, columns=df.keys())  # 本行有修订

def cal_mid_price(arrLike):
    price_bid = arrLike['BuyPriceQueue']
    price_ask = arrLike['SellPriceQueue']
    mid_price = (price_bid[0] + price_ask[0]) / 2
    return mid_price

