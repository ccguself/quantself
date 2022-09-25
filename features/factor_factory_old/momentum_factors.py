import numpy as np
import talib as ta
import pandas as pd
from .utils import ts_rank, scale, cal_mid_price

def speed_of_price(df, level:int =1, type: str='buy', interval_seconds=30, tick_interval_seconds=3):
    """
    speed_of_price
    """
    n_tick = interval_seconds // tick_interval_seconds
    bid_price = df["BuyPriceQueue"].apply(lambda x: x[level])
    ask_price = df["SellPriceQueue"].apply(lambda x: x[level])
    if type == 'buy':
        speed = bid_price.diff(n_tick) / interval_seconds * scale
    else:
        speed = ask_price.diff(n_tick) / interval_seconds * scale

    return speed

def px_to_ipov_premium_discount(df):
    """
    IOPV折价或溢价率
    """
    df['IOPVDiscount'] = np.log(df['IOPV'] / df['LastPx']) * 1000
    return df


def px_to_preclose_premium_discount(df):
    """
    相对昨收,实时折溢价率
    """
    df['PxChangeToPreCloseRealTime'] = np.log(
        df['LastPx'] / df['PreClosePx']) * 1000
    return df


def px_to_avgpx_premium_discount(df):
    """
    实时成交均价的折溢价率
    """
    df['PxChangeToAvgRealTime'] = np.log(
        df['LastPx'] / (df['TotalValueTrade'] / df['TotalVolumeTrade'])) * 1000
    return df


def px_to_high_premium_discount(df):
    """
    实时相对日内最高价的折溢价率
    """
    df['PxChangeToIntraDayHighest'] = np.log(
        df['LastPx'] / df['HighPx']) * 1000
    return df


def px_to_low_premium_discount(df):
    """
    实时相对日内最低价的折溢价率
    """
    df['PxChangeToIntraDayLowest'] = np.log(
        df['LastPx'] / df['LowPx']) * 1000
    return df


def realized_variance_ht(df, interval_seconds=30, tick_interval_seconds=3):
    """
    在 interval_seconds 内, 实现的var
    """
    n_tick = interval_seconds // tick_interval_seconds
    log_p = np.log(df['LastPx'])
    log_return = log_p.diff(1) * 1000
    realized_var = ta.SUM(log_return ** 2, n_tick)
    realized_var[realized_var <= 0] = 0
    df['ht_realized_var_{}'.format(interval_seconds)] = realized_var
    return df


def realized_variance_skewness_ht(df, interval_seconds=30, tick_interval_seconds=3):
    """
    在 interval_seconds 内, 实现的偏度
    """
    n_tick = interval_seconds // tick_interval_seconds
    log_p = np.log(df['LastPx'])
    log_return = log_p.diff(1) * 1000
    realized_var = ta.SUM(log_return ** 2, n_tick)
    realized_var[realized_var <= 0] = 0
    numerator = ta.SMA(np.sqrt(n_tick) *
                       ta.SUM(np.power(log_return, 3), n_tick))
    denominator = np.power(realized_var, 1.5)
    realized_skew = (numerator + 1e-10) / (denominator + 1e-10)
    df['ht_realized_skew_{}'.format(interval_seconds)] = realized_skew
    return df


def realized_variance_up_skewness_ht(df, interval_seconds=30, tick_interval_seconds=3):
    """
    在 interval_seconds 内, 实现的上行偏度
    """
    n_tick = interval_seconds // tick_interval_seconds
    log_p = np.log(df['LastPx'])
    log_return = log_p.diff(1) * 1000
    realized_var = ta.SUM(log_return ** 2, n_tick)
    realized_var[realized_var <= 0] = 0
    indicator_pos = (log_return > 0) * 1
    realized_var = ta.SUM(log_return ** 2, n_tick)
    numerator = ta.SMA(np.sqrt(n_tick) *
                       ta.SUM(np.power(log_return, 2), n_tick))
    realized_down = (numerator + 1e-10) / (realized_var + 1e-10)
    df['ht_realized_up_var_{}'.format(
        interval_seconds)] = indicator_pos * realized_down
    return df

def ret_trend_feature(df, interval_seconds=30, tick_interval_seconds=3):
    """
    A股收益的反转效应 -- 趋势强度，来自海通证券：高频量价因子在股票与期货中的表现
    """
    n_tick = interval_seconds // tick_interval_seconds
    df['trendStrength_{}'.format(interval_seconds)] = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (ta.SUM(
        abs(df['LastPx'] - df['LastPx'].shift(1)), n_tick) + 1e-10)
    return df


def ret_trend_feature_abs(df, interval_seconds=30, tick_interval_seconds=3):
    """
    A股收益的反转效应_abs -- 趋势强度，来自海通证券：高频量价因子在股票与期货中的表现
    """
    n_tick = interval_seconds // tick_interval_seconds
    df['trendStrength_abs_{}'.format(interval_seconds)] = abs((df['LastPx'] - df['LastPx'].shift(n_tick)) + 1e-10) / (ta.SUM(
        abs(df['LastPx'] - df['LastPx'].shift(1)), n_tick) + 1e-10)
    return df

def fifth_moment(df, interval_seconds=30, tick_interval_seconds=3):
    """
    收益率序列5阶矩 - 华泰-动量增强（行业轮动系列）
    """
    num_ticks = interval_seconds // tick_interval_seconds
    return_pct = df["LastPx"].pct_change() * scale
    high_moment = ta.SMA(return_pct ** 5, timeperiod=num_ticks)
    high_moment_ema = high_moment.ewm(span=num_ticks, adjust=False).mean()
    return high_moment_ema

def px_sharpe_ratio(df, interval_seconds=300, tick_interval_seconds=3):
    """
    px_sharpe_ratio	收益率 / 波动率 - 华泰-动量增强（行业轮动系列）
    """
    n_tick = interval_seconds // tick_interval_seconds
    return_pct = df["LastPx"].pct_change() * scale
    mean_return = ta.SMA(return_pct, n_tick)
    std_return = ta.STDDEV(return_pct, n_tick)
    px_shape_ratio = (mean_return + 1e-10) / (std_return + 1e-10)
    return px_shape_ratio


def path_adjusted_momentum(df, interval_seconds=30, tick_interval_seconds=3):
    """
    path_adjusted_momentum 路径调整后的动量 - 华泰-动量增强（行业轮动系列）
    """
    n_tick = interval_seconds // tick_interval_seconds
    last_price = df["LastPx"]
    last_price_shift = df['LastPx'].shift(n_tick)
    momentum = ((last_price_shift + 1e-10) / (last_price + 1e-10) - 1)
    momentum.replace(np.inf, 0, inplace=True)
    distance = ta.SUM(abs(last_price.pct_change()), n_tick)
    path_adjusted_momentum = (momentum + 1e-10) / (distance + 1e-10) * scale
    return path_adjusted_momentum


def max_pct(df, interval_seconds=30, tick_interval_seconds=3):
    """
    tick_interval_seconds 内的最大涨幅 - 华泰-动量增强（行业轮动系列）
    """
    n_tick = interval_seconds // tick_interval_seconds
    return_pct = df["LastPx"].pct_change() * scale
    max_pct = ta.MAX(return_pct, n_tick)
    return max_pct

# 动量指标mtm
def tech_indicator_for_mtm(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    '''
    MTM线　:当日收盘价与N日前的收盘价的差；
    MTMMA线:对上面的差值求m日移动平均；
    参数：N 间隔天数，也是求移动平均的天数，一般取6用法： 
    1.MTM从下向上突破MTMMA，买入信号；
    2.MTM从上向下跌破MTMMA，卖出信号； 
    3.股价续创新高，而MTM未配合上升，意味上涨动力减弱； 
    4.股价续创新低，而MTM未配合下降，意味下跌动力减弱； 
    5.股价与MTM在低位同步上升，将有反弹行情；反之，从高位同步下降，将有回落走势。
    '''

    df['tech_indicator_for_mtm_{}'.format(interval_seconds)] = ta.SMA(df['LastPx'] - df['LastPx'].shift(n_tick),
                                                                      n_tick * 2)
    return df


def middle_movei(df, interval_seconds=60,  tick_interval_seconds=3):
    # 对于中间价计算滑动平均结果，并计算当前中间价与滑动平滑中间价的差异
    n_tick = interval_seconds // tick_interval_seconds
    df['mid_price'] = df.apply(cal_mid_price, axis=1)
    mid_price_sma = ta.SMA(df['mid_price'], timeperiod = n_tick)
    df['MiddleMove_{}'.format()] = (df['mid'] - mid_price_sma) / mid_price_sma * scale
    del (df['mid_price'])
    return df
