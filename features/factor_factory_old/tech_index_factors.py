import numpy as np
import talib as ta
import pandas as pd
from .utils import scale, ts_rank, ts_cov, GetRunTime, cal_mid_price
from pyfinance.ols import PandasRollingOLS
from tsfresh.feature_extraction import feature_calculators


def force_index(df, interval_seconds=30, tick_interval_seconds=3):
    """
    劲道指数 Force Index
    """
    n_tick = interval_seconds // tick_interval_seconds
    return ta.EMA(df['LastPx'].diff(1) * df['TotalVolumeTrade'].diff(1), n_tick)


def rsi(df, interval_seconds=30, tick_interval_seconds=3):
    """
    RSI
    """
    n_tick = interval_seconds // tick_interval_seconds
    df['Rsi_{}'.format(interval_seconds)] = ta.RSI(
        df['LastPx'], timeperiod=n_tick)
    return df


def roc(df, interval_seconds=30, tick_interval_seconds=3):
    """
    ROC
    """
    n_tick = interval_seconds // tick_interval_seconds
    df['Roc_{}'.format(interval_seconds)] = ta.ROC(
        df['LastPx'], timeperiod=n_tick)
    return df



def cmo(df, interval_seconds=30, tick_interval_seconds=3):
    """
    CMO
    """
    n_tick = interval_seconds // tick_interval_seconds
    df['Cmo_{}'.format(interval_seconds)] = ta.CMO(
        df['LastPx'], timeperiod=n_tick)
    return


def macd(df):
    """
    MACD
    """
    Diff, Dea, MACD = ta.MACD(
        df['LastPx'], fastperiod=6, slowperiod=12, signalperiod=15)
    return Diff, Dea, MACD


def dpo(df, interval_seconds=30, tick_interval_seconds=3):
    """
    DPO
    """
    n_tick = interval_seconds // tick_interval_seconds
    temp = ta.SMA(df['LastPx'], n_tick)
    df['DPO_{}'.format(interval_seconds)] = df['LastPx'] - \
        temp.shift(n_tick // 2 + 1)
    return df


def bull_power(df, interval_seconds=30, tick_interval_seconds=3):
    '''
    最高价突破
    '''
    n_tick = interval_seconds // tick_interval_seconds
    df['Bull_Power_{}'.format(interval_seconds)
       ] = df['HighPx'] - ta.EMA(df['LastPx'], n_tick)
    return df


def bear_power(df, interval_seconds=30, tick_interval_seconds=3):
    '''
    最低价突破
    '''
    n_tick = interval_seconds // tick_interval_seconds
    df['Bear_Power_{}'.format(interval_seconds)
       ] = df['LowPx'] - ta.EMA(df['LastPx'], n_tick)
    return df


def po(df, interval_seconds=30, tick_interval_seconds=3):
    """
    PO
    """
    n_tick = interval_seconds // tick_interval_seconds
    long_price = ta.EMA(df['LastPx'], n_tick // 3)
    short_price = ta.EMA(df['LastPx'], n_tick)
    df['PO_{}'.format(interval_seconds)] = (
        short_price - long_price) / long_price * 1000
    return df


def pos(df, interval_seconds=30, tick_interval_seconds=3):
    """
    POS
    """
    n_tick = interval_seconds // tick_interval_seconds
    temp = (df['LastPx'] - df['LastPx'].shift(n_tick)) / \
        (df['LastPx'].shift(n_tick))
    df['POS_{}'.format(interval_seconds)] = (
        temp - ta.MIN(temp, n_tick)) / (ta.MAX(temp, n_tick) - ta.MIN(temp, n_tick) + 1e-10)
    return df


def MPB(df):
    df['mid_price'] = df.apply(cal_mid_price, axis=1)
    df['tick_prc'] = df['mid_price']
    prc = df['TotalValueTrade'] / df['TotalVolumeTrade']
    df.loc[df['TotalVolumeTrade'] > 0, "tick_prc"] = prc[df['TotalVolumeTrade'] > 0]
    df['MPB'] = df['tick_prc'] - df['mid_price']
    del df['mid_price']
    return df



def trend_strength(df, k=10):
    df['mid_price'] = df.apply(cal_mid_price, axis=1)
    df['diff_mid'] = df['mid_price'] - df['mid_price'].shift(1)
    rolling_sum1 = df['diff_mid'].rolling(window=k).sum()
    rolling_sum2 = abs(df['diff_mid']).rolling(window=k).sum()
    df['trend_strength'] = rolling_sum1 / rolling_sum2
    del df['diff_mid']
    del df['mid_price']
    return df


def tech_indicator_for_cmf(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    '''
    MFV = (2CLOSE - LOW -HIGH)/(HIGH-LOW)*VOLUME
    CMF = SUM(MFV,NDAY)/SUM(V,NDAY)
    '''

    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    mfv = (2 * df['LastPx'] - df['LowPx'] - df['HighPx'] + 1e-10) / (df['HighPx'] - df['LowPx'] + 1e-10) * volume

    df['tech_indicator_for_cmf_{}'.format(interval_seconds)] = (ta.SMA(mfv, n_tick) + 1e-10) / (
                ta.SMA(volume, n_tick) + 1e-10)
    return df


def tech_indicator_for_wr(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    """
    WR 威廉指标
    1.WR波动于0 - 100，100置于顶部，0置于底部。
    2.本指标以50为中轴线，高于50视为股价转强；低于50视为股价转弱
    3.本指标高于20后再度向下跌破20，卖出；低于80后再度向上突破80，买进。
    4.WR连续触底3 - 4次，股价向下反转机率大；连续触顶3 - 4次，股价向上反转机率大。
    """

    df['tech_indicator_for_wr_{}'.format(interval_seconds)] = (ta.MAX(df['HighPx'], n_tick) - df['LastPx'] + 1e-10) / (
                ta.MAX(df['HighPx'], n_tick) - ta.MAX(df['LowPx'], n_tick) + 1e-10)
    return df

# 商品路径指标cci
def tech_indicator_for_cci(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    '''
    CCI指标，CCI（N日）=（TP－MA）÷MD÷0.015
    其中，TP=（最高价+最低价+收盘价）÷3
    MA=近N日收盘价的累计之和÷N
    MD=近N日（MA－收盘价）的累计之和÷N
    0.015为计算系数，N为计算周期
    1.CCI 为正值时，视为多头市场；为负值时，视为空头市场； 
    2.常态行情时，CCI 波动于±100 的间；强势行情，CCI 会超出±100 ；
    3.CCI>100 时，买进，直到CCI<100 时，卖出；
    4.CCI<-100 时，放空，直到CCI>-100 时，回补。
    '''
    tp = (df['HighPx'] + df['LowPx'] + df['LastPx']) / 3
    ma = ta.SMA(df['LastPx'], n_tick)
    md = ta.SMA(ma - df['LastPx'], n_tick)
    df['tech_indicator_for_cci_{}'.format(interval_seconds)] = (tp - ma + 1e-10) / (md + 1e-10)
    return df


# 统计n_tick之内的上涨超过某个阈值的次数
def up_number(df, interval_seconds=30, tick_interval_seconds=3, nth=0.0004):
    n_tick = interval_seconds // tick_interval_seconds

    ret = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (df['LastPx'] + 1e-10)
    ret_up = (ret > nth) * 1

    df['up_number_{}'.format(interval_seconds)] = ta.SUM(ret_up, n_tick)
    return df


# 统计n_tick之内的下跌超过某个阈值的次数
def down_number(df, interval_seconds=30, tick_interval_seconds=3, nth=0.0004):
    n_tick = interval_seconds // tick_interval_seconds

    ret = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (df['LastPx'] + 1e-10)
    ret_down = (ret < -1 * nth) * 1

    df['down_number_{}'.format(interval_seconds)] = ta.SUM(ret_down, n_tick)
    return df


# 统计n_tick之内的上涨超过某个阈值即下跌超过某个阈值的比率
def up_down_number_ratio(df, interval_seconds=30, tick_interval_seconds=3, nth=0.0004):
    n_tick = interval_seconds // tick_interval_seconds
    ret = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (df['LastPx'] + 1e-10)
    ret_up = (ret > nth) * 1
    ret_down = (ret < -1 * nth) * 1
    df['up_down_number_ratio_{}'.format(interval_seconds)] = (ta.SUM(ret_up, n_tick) + 1e-10) / (
                ta.SUM(ret_down, n_tick) + 1e-10)
    return df


# 统计前2*tick内涨幅超过某个阈值的次数 与 前tick内涨幅超过某个阈值的次数 的 比率
def up_number_ratio_2tick(df, interval_seconds=30, tick_interval_seconds=3, nth=0.0004):
    n_tick = interval_seconds // tick_interval_seconds

    ret = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (df['LastPx'] + 1e-10)
    ret_up = (ret > nth) * 1
    # ret_down = (ret < -1 * nth) * 1

    df['up_number_ratio_2tick_{}'.format(interval_seconds)] = (ta.SUM(ret_up, n_tick * 2) + 1e-10) / (
                ta.SUM(ret_up, n_tick) + 1e-10)
    return df


# 统计前2*tick内跌幅超过某个阈值的次数 与 前tick内跌幅超过某个阈值的次数 的 比率
def down_number_ratio_2tick(df, interval_seconds=30, tick_interval_seconds=3, nth=0.0004):
    n_tick = interval_seconds // tick_interval_seconds

    ret = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (df['LastPx'] + 1e-10)
    # ret_up = (ret > nth) * 1
    ret_down = (ret < -1 * nth) * 1

    df['down_number_ratio_2tick_{}'.format(interval_seconds)] = (ta.SUM(ret_down, n_tick * 2) + 1e-10) / (
                ta.SUM(ret_down, n_tick) + 1e-10)
    return df


# 统计前2*tick内涨幅超过某个阈值的次数 与 前tick内涨幅超过某个阈值的次数 的 比率 * volume ratio
def up_number_ratio_2tick_volraito(df, interval_seconds=30, tick_interval_seconds=3, nth=0.0004):
    n_tick = interval_seconds // tick_interval_seconds

    ret = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (df['LastPx'] + 1e-10)
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    ret_up = (ret > nth) * 1
    # ret_down = (ret < -1 * nth) * 1
    volume_ratio = (ta.SUM(volume, 2 * n_tick) + 1e-10) / (ta.SUM(volume, n_tick) + 1e-10)

    df['up_number_ratio_2tick_volraito_{}'.format(interval_seconds)] = (ta.SUM(ret_up, n_tick * 2) + 1e-10) / (
                ta.SUM(ret_up, n_tick) + 1e-10) * volume_ratio
    return df


# 统计前2*tick内涨幅超过某个阈值的次数 与 前tick内涨幅超过某个阈值的次数 的 比率 * volume ratio
def up_number_ratio_2tick_volraito_2tick(df, interval_seconds=30, tick_interval_seconds=3, nth=0.0004, vth=0.001):
    n_tick = interval_seconds // tick_interval_seconds

    ret = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (df['LastPx'] + 1e-10)
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    ret_up = (ret > nth) * 1
    # ret_down = (ret < -1 * nth) * 1
    volume_ratio = (volume + 1e-10) / (ta.SMA(volume, n_tick) + 1e-10)
    vol_up = (volume_ratio > vth) * 1

    df['up_number_ratio_2tick_volraito_{}'.format(interval_seconds)] = (ta.SUM(ret_up, n_tick * 2)
                                                                        + 1e-10) / (ta.SUM(ret_up, n_tick) + 1e-10) * (
                                                                                   ta.SUM(vol_up,
                                                                                          n_tick * 2) + 1e-10) / (
                                                                                   ta.SUM(vol_up, n_tick) + 1e-10)
    return df



def tech_indicator_for_b3612(df, interval_seconds=30, tick_interval_seconds=3):
    '''
    乖离值围绕多空平衡点零上下波动，正数达到某个程度无法再往上升时，
    是卖出时机；反之，是买进时机。
    多头走势中，行情回档多半在三减六日乖离达到零附近获得支撑，即使跌破，也很快能够拉回
    '''
    n_tick = interval_seconds // tick_interval_seconds
    b36 = ta.SMA(df['LastPx'], n_tick) - ta.SMA(df['LastPx'], n_tick * 2)
    b612 = ta.SMA(df['LastPx'], n_tick * 2) - ta.SMA(df['LastPx'], n_tick * 3)

    ret = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (df['LastPx'] + 1e-10)
    ret_up = (ret > 0) * 1
    ret_down = (ret < 0) * 1

    df['tech_indicator_for_b3612_{}'.format(interval_seconds)] = b36 * ret_up - b612 * ret_down

    return df


def tech_indicator_for_osc(df, interval_seconds=30, tick_interval_seconds=3):
    '''
    osc摆动线
    当震荡点大于0且股价趋势仍属上升时，为多头走势，反之当震荡点小于0且股价趋势
    为下跌是为空头走势。 
    '''
    n_tick = interval_seconds // tick_interval_seconds
    osc = df['LastPx'] - ta.SMA(df['LastPx'], n_tick)

    ret = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (df['LastPx'] + 1e-10)
    ret_up = (ret > 0) * 1
    ret_down = (ret < 0) * 1

    df['tech_indicator_for_osc_{}'.format(interval_seconds)] = osc * ret_up - osc * ret_down

    return df

    '''
    psy心理线,心理线是一种建立在研究投资人心理趋向基础上，
    将某段时间内投资者倾向买方还是卖方的心理与事实转化为数值，形成人气指标，做为买卖股票的参数。 
    这里可对上述涨跌幅超过某个阈值的反转因子将阈值设为0，即为psy心理线因子
    '''


def tech_indicator_for_bias(df, interval_seconds=30, tick_interval_seconds=3):
    '''
    是移动平均原理派生的一项技术指标，
    其功能主要是通过测算股价在波动过程中与移动平均线出现偏离的程度，
    从而得出股价在剧烈波动时因偏离移动平均趋势而造成可能的回档或反弹，
    以及股价在正常波动范围内移动而形成继续原有势的可信度。 
    '''
    n_tick = interval_seconds // tick_interval_seconds

    df['tech_indicator_for_bias_{}'.format(interval_seconds)] = (df['LastPx'] - ta.SMA(df['LastPx'],
                                                                                       n_tick) + 1e-10) / (
                                                                            ta.SMA(df['LastPx'], n_tick) + 1e-10)

    return df


def tech_indicator_for_sar(df, interval_seconds=30, tick_interval_seconds=3):
    '''
    停损点转向又称抛物线转向，因其坐标点形成的连线呈抛物线状而得名。
    它通过设置一个极点值（4日内最高价或最低价），当极点值与行情价格交叉时，
    提醒投资者及时由多转空，或由空转多。该技术指标得出的买卖结论是最明确的。
    '''
    n_tick = interval_seconds // tick_interval_seconds
    px_max = ta.MAX(df['LastPx'], n_tick)
    ret = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (df['LastPx'] + 1e-10)
    ret_up = (ret > 0) * 1
    ret_down = (ret < 0) * 1

    df['tech_indicator_for_sar_{}'.format(interval_seconds)] = (df['LastPx'] - px_max) * ret_up - (
                df['LastPx'] - px_max) * ret_down

    return df


def tech_indicator_for_expma(df, interval_seconds=30, tick_interval_seconds=3):
    '''
    1、当短期指数平均数由下往上穿过长期平均数时为买进讯号， 
    2、当短期指数平均数幅上往下空过长期平均数时为卖出讯号。 
    3、股价由下往上碰触EXPMA时，很容易遭遇大压力回档。 
    4、股价由上往下碰触EXPMA时，很容易遭遇大支撑反弹。
    该指标以交叉为主要讯号。该指标可以随股价的快速移动，立即调整方向，有效地解决讯号落后的问题，但该指标在盘整行情中不适用。
    '''
    n_tick = interval_seconds // tick_interval_seconds
    expma1 = (df['LastPx'] - ta.SMA(df['LastPx'], n_tick).shift(1)) * 2.0 / (n_tick + 1) + ta.SMA(df['LastPx'],
                                                                                                  n_tick).shift(1)
    expma2 = (df['LastPx'] - ta.SMA(df['LastPx'], n_tick * 2).shift(1)) * 2.0 / (n_tick * 2 + 1) + ta.SMA(df['LastPx'],
                                                                                                          n_tick).shift(
        1)
    ret = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (df['LastPx'] + 1e-10)

    ret_up = (ret > 0) * 1
    ret_down = (ret < 0) * 1

    df['tech_indicator_for_expma_{}'.format(interval_seconds)] = expma1 * ret_up + expma2 * ret_down

    return df


@GetRunTime
def get_abs_energy(df, interval_seconds=300, tick_interval_seconds=3):
    """Returns the absolute energy of the time series which is the sum over the squared values

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.

    Returns:
        np.array: The sum over the squared values
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_abs_energy(x):
        return feature_calculators.abs_energy(x)

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_abs_energy).values
    df.drop(columns='pct_change', inplace=True)
    return ans

@GetRunTime
def get_absolute_sum_of_changes(df, interval_seconds=300, tick_interval_seconds=3):
    """Returns the sum over the absolute value of consecutive changes in the series x

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.

    Returns:
        np.array: The sum over the absolute value of consecutive changes in the series x
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_absolute_sum_of_changes(x):
        return feature_calculators.absolute_sum_of_changes(x)

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_absolute_sum_of_changes).values
    df.drop(columns='pct_change', inplace=True)

    return ans

@GetRunTime
def get_agg_autocorrelation(df, interval_seconds=300, tick_interval_seconds=3, lag_max=10, method="mean"):
    """ Calculates the value of an aggregation function(mean) over the autocorrelation R(l) for different lags.

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.
        lag_max(int): 进行自相关的线性拟合，最大延迟的数值

    Returns:
        np.array: The aggregation function over the autocorrelation R(l) for different lags.
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_agg_autocorrelation(x):
        return feature_calculators.agg_autocorrelation(x, [{"f_agg": method, "maxlag": lag_max}])[0][1]

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_agg_autocorrelation).values
    df.drop(columns='pct_change', inplace=True)
    return ans

@GetRunTime
def get_agg_linear_trend(
    df, interval_seconds=300, tick_interval_seconds=3, chunk=20, agg_method="mean", attr_method="slope"
):
    """Calculates a linear least-squares regression for values of the time series

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.
        chunk (int, optional): 多少个观测点做一次回归. Defaults to 20.
        agg_method (str, optional): [description]. Defaults to "mean".
        attr_method (str, optional): [description]. Defaults to "slope".

    Returns:
        np.array: The aggregation function over the ols values.
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_agg_linear_trend(x):
        return list(
            feature_calculators.agg_linear_trend(x, [{"attr": attr_method, "chunk_len": chunk, "f_agg": agg_method}])
        )[0][1]

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_agg_linear_trend).values
    df.drop(columns='pct_change', inplace=True)
    return ans

# 注意：此处filter_level可以根据不同的开仓选择进行修改（千一，千二或其他）
@GetRunTime
def get_approximate_entropy(df, interval_seconds=300, tick_interval_seconds=3, compared_length=3, filter_level=0.002):
    """Implements a vectorized Approximate entropy algorithm.

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.
        compared_length (int, optional): [description]. Defaults to 3.
        filter_level (float, optional): [description]. Defaults to 0.002.

    Returns:
        np.array: The appoximate entropy
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_approximate_entropy(x):
        return feature_calculators.approximate_entropy(x, m=compared_length, r=filter_level)


    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_approximate_entropy).values
    df.drop(columns='pct_change', inplace=True)

    return ans


# 注意：可以多输出几个out_ar
@GetRunTime
def get_ar_coefficient(df, interval_seconds=300, tick_interval_seconds=3, out_ar=1, max_ar=5):
    """This feature calculator fits the unconditional maximum likelihood of an autoregressive AR(k) process.

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.
        out_ar (int, optional): [description]. Defaults to 1.
        max_ar (int, optional): [description]. Defaults to 5.

    Returns:
        np.array: The ar coefficient
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_ar_coefficient(x):
        return feature_calculators.ar_coefficient(x, [{"coeff": out_ar, "k": max_ar}])[0][1]

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_ar_coefficient).values
    df.drop(columns='pct_change', inplace=True)

    return ans

@GetRunTime
def get_augmented_dickey_fuller(
    df, interval_seconds=300, tick_interval_seconds=3, attr_method="pvalue", autolag_method="AIC"
):
    """The Augmented Dickey-Fuller test is a hypothesis test

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.
        attr_method (str, optional): [description]. Defaults to "pvalue".
        autolag_method (str, optional): [description]. Defaults to "AIC".

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_augmented_dickey_fuller(x):
        return feature_calculators.augmented_dickey_fuller(x, [{"attr": attr_method, "autolag": autolag_method}])[0][1]

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_augmented_dickey_fuller).values
    df.drop(columns='pct_change', inplace=True)

    return ans

@GetRunTime
def get_binned_entropy(df, interval_seconds=300, tick_interval_seconds=3, num_bins=10):
    """First bins the values of x into max_bins equidistant bins.

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.
        num_bins (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_binned_entropy(x):
        return feature_calculators.binned_entropy(x, max_bins=num_bins)
    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_binned_entropy).values
    df.drop(columns='pct_change', inplace=True)

    return ans


# 注意：可以选择不同的num_lag
@GetRunTime
def get_c3(df, interval_seconds=300, tick_interval_seconds=3, num_lag=1):
    """具体计算方法参见tsfresh文档

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.
        num_lag (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_c3(x):
        return feature_calculators.c3(x, lag=num_lag)

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_c3).values
    df.drop(columns='pct_change', inplace=True)

    return ans



@GetRunTime
def get_cid_ce(df, interval_seconds=300, tick_interval_seconds=3):
    """This function calculator is an estimate for a time series complexity

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_cid_ce(x):
        return feature_calculators.cid_ce(x, normalize=True)
    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_cid_ce).values
    df.drop(columns='pct_change', inplace=True)
    return ans


# Calculates a Continuous wavelet transform for the Ricker wavelet, also known as the “Mexican hat wavelet”
# def get_cwt_coefficients(df, interval_seconds=300, tick_interval_seconds=3):


# 注意：可以使用不同的segment_focus
@GetRunTime
def get_energy_ratio_by_chunks(df, interval_seconds=300, tick_interval_seconds=3, num_segment=10, segment_focus=1):
    """Calculates the sum of squares of chunk i out of N chunks expressed as a ratio with the sum of squares over the whole series.

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.
        num_segment (int, optional): 切片个数. Defaults to 10.
        segment_focus (int, optional): 返回的值来自哪个切片. Defaults to 1.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_energy_ratio_by_chunks(x):
        return feature_calculators.energy_ratio_by_chunks(
            x, [{"num_segments": num_segment, "segment_focus": segment_focus}]
        )[0][1]

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_energy_ratio_by_chunks).values
    df.drop(columns='pct_change', inplace=True)

    return ans


# 分别选择返回不同的矩信息
@GetRunTime
def get_fft_aggregated(df, interval_seconds=300, tick_interval_seconds=3, method='centroid'):
    """Returns the spectral centroid (mean), variance, skew, and kurtosis of the absolute fourier transform spectrum.

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.
        num_segment (int, optional): 切片个数. Defaults to 10.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_fft_aggregated_by_method(x):
        return list(feature_calculators.fft_aggregated(x, [{"aggtype": method}]))[0][1]

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_fft_aggregated_by_method).values
    df.drop(columns='pct_change', inplace=True)

    return ans





# Calculates the fourier coefficients of the one-dimensional discrete Fourier Transform for real input by fast fourier transformation algorithm

@GetRunTime
def get_first_location_of_maximum(df, interval_seconds=300, tick_interval_seconds=3):
    """Returns the first location of the maximum value of x. The position is calculated relatively to the length of x.

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_first_location_of_maximum(x):
        return feature_calculators.first_location_of_maximum(x)

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_first_location_of_maximum).values
    df.drop(columns='pct_change', inplace=True)


    return ans

@GetRunTime
def get_first_location_of_minimum(df, interval_seconds=300, tick_interval_seconds=3):
    """Returns the first location of the minimum value of x. The position is calculated relatively to the length of x.

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_first_location_of_minimum(x):
        return feature_calculators.first_location_of_minimum(x)

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_first_location_of_minimum).values
    df.drop(columns='pct_change', inplace=True)

    return ans


# Calculate the binned entropy of the power spectral density of the time series (using the welch method).
# def get_


# Coefficients of polynomial h(x), which has been fitted to the deterministic dynamics of Langevin model
# def get_friedrich_coefficients(df, interval_seconds=300, tick_interval_seconds=3):


#
@GetRunTime
def get_longest_strike_above_mean(df, interval_seconds=300, tick_interval_seconds=3):
    """longest_strike_above_mean

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_longest_strike_above_mean(x):
        return feature_calculators.longest_strike_above_mean(x)

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_longest_strike_above_mean).values
    df.drop(columns='pct_change', inplace=True)

    return ans


# longest_strike_below_mean
@GetRunTime
def get_longest_strike_below_mean(df, interval_seconds=300, tick_interval_seconds=3):
    """longest_strike_above_mean

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_longest_strike_below_mean(x):
        return feature_calculators.longest_strike_below_mean(x)


    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_longest_strike_below_mean).values
    df.drop(columns='pct_change', inplace=True)
    return ans

# max_langevin_fixed_point
# def

# 注意：可以根据开仓方式进行调整m
@GetRunTime
def get_number_crossing_m(df, interval_seconds=300, tick_interval_seconds=3, m=0):
    """number_crossing_m

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.
        m (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_number_crossing_m(x):
        return feature_calculators.number_crossing_m(x, m)

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_number_crossing_m).values
    df.drop(columns='pct_change', inplace=True)
    return ans


@GetRunTime
def get_number_cwt_peaks(df, interval_seconds=300, tick_interval_seconds=3, n=10):
    """searches for different peaks in x. To do so, x is smoothed by a ricker wavelet and for widths ranging from 1 to n.

    Args:
        df (pd.DataFrame): 原始特征矩阵（由于并行化处理，df包含的是单日数据）
        interval_seconds (int, optional): 表示多少秒之前的市场状态. Defaults to 300s.
        tick_interval_seconds (int, optional): 多少秒切一次样本. Defaults to 3s.
        n (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_number_cwt_peaks(x):
        return feature_calculators.number_cwt_peaks(x, n)

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_number_cwt_peaks).values
    df.drop(columns='pct_change', inplace=True)
    return ans



# 注意：可以使用不同的lag
@GetRunTime
def get_partial_autocorrelation(df, interval_seconds=300, tick_interval_seconds=3, lag=1):
    """the value of the partial autocorrelation function at the given lag.

    Args:
        df ([type]): [description]
        interval_seconds (int, optional): [description]. Defaults to 300.
        tick_interval_seconds (int, optional): [description]. Defaults to 3.
        lag (int, optional): [description]. Defaults to 1.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_partial_autocorrelation(x):
        return feature_calculators.partial_autocorrelation(x, [{"lag": lag}])[0][1]

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_partial_autocorrelation).values
    df.drop(columns='pct_change', inplace=True)
    return ans


@GetRunTime
def get_sample_entropy(df, interval_seconds=300, tick_interval_seconds=3):
    """Calculate and return sample entropy of x.

    Args:
        df ([type]): [description]
        interval_seconds (int, optional): [description]. Defaults to 300.
        tick_interval_seconds (int, optional): [description]. Defaults to 3.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_sample_entropy(x):
        return feature_calculators.sample_entropy(x)

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_sample_entropy).values
    df.drop(columns='pct_change', inplace=True)
    return ans

@GetRunTime
def get_spkt_welch_density(df, interval_seconds=300, tick_interval_seconds=3):
    """This feature calculator estimates the cross power spectral density of the time series x at different frequencies.

    Args:
        df ([type]): [description]
        interval_seconds (int, optional): [description]. Defaults to 300.
        tick_interval_seconds (int, optional): [description]. Defaults to 3.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_spkt_welch_density(x):
        return list(feature_calculators.spkt_welch_density(x, [{"coeff": 1}]))[0][1]

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_spkt_welch_density).values
    df.drop(columns='pct_change', inplace=True)
    return ans


# 注意：此处lag可以选择（可根据预测的窗口进行选择）
@GetRunTime
def get_time_reversal_asymmetry_statistic(df, interval_seconds=300, tick_interval_seconds=3, num_lag=10):
    """time_reversal_asymmetry_statistic

    Args:
        df ([type]): [description]
        interval_seconds (int, optional): [description]. Defaults to 300.
        tick_interval_seconds (int, optional): [description]. Defaults to 3.
        num_lag (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    n_tick = interval_seconds // tick_interval_seconds

    def cal_time_reversal_asymmetry_statistic(x):
        return feature_calculators.time_reversal_asymmetry_statistic(x, lag=num_lag)

    df["pct_change"] = df["LastPx"].pct_change().fillna(0)
    ans = df["pct_change"].rolling(n_tick).apply(cal_time_reversal_asymmetry_statistic).values
    df.drop(columns='pct_change', inplace=True)
    return ans


def information_dispersion(df, interval_seconds=30, tick_interval_seconds=3):
    """
    信息离散率 - 华泰-动量增强（行业轮动系列）
    """
    n_tick = interval_seconds // tick_interval_seconds
    last_price = df["LastPx"]
    last_price_shift = df['LastPx'].shift(n_tick)
    pct_change = ((last_price_shift + 1e-10) / (last_price + 1e-10) - 1)
    pct_change = pct_change.replace(np.nan, 0)
    block_return_symbol = np.sign(pct_change)
    block = last_price.pct_change()
    num_positive = ta.SUM((block > 0) * 1, n_tick)
    num_negative = ta.SUM((block < 0) * 1, n_tick)
    information_dispersion = block_return_symbol * \
        (num_positive - num_negative)
    return information_dispersion