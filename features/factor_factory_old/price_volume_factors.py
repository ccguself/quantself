import numpy as np
import talib as ta
import pandas as pd
from .utils import ts_rank, scale


def order_dispersion(df: pd.DataFrame):
    """
    买卖盘的离差程度
    source: https://mp.weixin.qq.com/s?__biz=MzkxMDE2NDc2Mw==&mid=2247484177&amp;idx=1&amp;sn=8e9f82e9b8e3ed4d5d15f774555e80d1&source=41#wechat_redirect
    """
    def get_order_dispersion(arrLike):
        vol_bid = arrLike["BuyOrderQtyQueue"]
        vol_ask = arrLike["SellOrderQtyQueue"]
        price_bid = arrLike['BuyPriceQueue']
        price_ask = arrLike['SellPriceQueue']
        ds1 = (np.sum(np.diff(price_bid, n=1) * vol_bid[0:-1]) + 1e-10) / ((np.sum(vol_bid[0:-1])) + 1e-10)
        ds2 = (np.sum(np.diff(price_ask, n=1) * vol_ask[0:-1]) + 1e-10) / ((np.sum(vol_ask[0:-1])) + 1e-10)
        return (ds1 + ds2) / 2

    df['order_dispersion'] = df.apply(get_order_dispersion, axis=1)
    return

def average_bid_slopes(df: pd.DataFrame):
    """
    买盘的斜率
    source: https://mp.weixin.qq.com/s?__biz=MzkxMDE2NDc2Mw==&mid=2247484177&amp;idx=1&amp;sn=8e9f82e9b8e3ed4d5d15f774555e80d1&source=41#wechat_redirect
    """
    def get_bid_slopes(arrLike):
        vol_bid = arrLike["BuyOrderQtyQueue"]
        vol_ask = arrLike["SellOrderQtyQueue"]
        price_bid = arrLike['BuyPriceQueue']
        price_ask = arrLike['SellPriceQueue']
        m1 = (price_bid[0] + price_ask[0]) / 2

        lp2 = np.sum(((vol_bid[0:-1] + 1e-10) / (vol_bid[1:] + 1e-10) - 1 + 1e-10) / ((price_bid[0:-1] + 1e-10) / (price_bid[1:] + 1e-10) - 1 + 1e-10))
        lp1 = (vol_bid[0] + 1e-10) / (m1 / price_bid[0] - 1 + 1e-10)
        slopes_bid = 1 / len(vol_bid) * (lp1 + lp2)
        return slopes_bid
    df['average_bid_slopes'] = df.apply(get_bid_slopes, axis=1)
    return


def average_ask_slopes(df: pd.DataFrame):
    """
    卖盘的斜率
    source: https://mp.weixin.qq.com/s?__biz=MzkxMDE2NDc2Mw==&mid=2247484177&amp;idx=1&amp;sn=8e9f82e9b8e3ed4d5d15f774555e80d1&source=41#wechat_redirect
    """
    def get_ask_slopes(arrLike):
        vol_bid = arrLike["BuyOrderQtyQueue"]
        vol_ask = arrLike["SellOrderQtyQueue"]
        price_bid = arrLike['BuyPriceQueue']
        price_ask = arrLike['SellPriceQueue']
        m1 = (price_bid[0] + price_ask[0]) / 2
        lp2 = np.sum(((vol_ask[1:] + 1e-10) / (vol_ask[0:-1] + 1e-10) - 1 + 1e-10) / ((price_ask[1:] + 1e-10) / (price_ask[0:-1] + 1e-10) - 1 + 1e-10))
        lp1 = (vol_ask[0] + 1e-10) / ( price_ask[0] / m1 - 1 + 1e-10)
        slopes_ask = 1 / len(vol_ask) * (lp1 + lp2)
        return slopes_ask
    df['average_ask_slopes'] = df.apply(get_ask_slopes, axis=1)
    return


def px_change_ratio(df, interval_seconds=30, tick_interval_seconds=3):
    """
    在 interval_seconds 内的涨跌幅
    """
    n_tick = interval_seconds // tick_interval_seconds
    return df['LastPx'].diff(n_tick) / df['LastPx'].shift(n_tick) * 1000



def cor_px_vol(df, interval_seconds=30, tick_interval_seconds=3):
    """
    在 interval_seconds 内的量价相关性
    """
    n_tick = interval_seconds // tick_interval_seconds
    return ta.CORREL(df['LastPx'], df['TotalVolumeTrade'].diff(1), n_tick)


def px_vol_corr_slope(df, interval_seconds=30, tick_interval_seconds=3):
    """
    在 interval_seconds 内, 量价相关性的斜率
    """
    n_tick = interval_seconds // tick_interval_seconds
    pv_cor = ta.CORREL(df['LastPx'], df['TotalVolumeTrade'].diff(1), n_tick)
    pv_cor_slope = ta.LINEARREG_SLOPE(pv_cor, n_tick)
    df['cor_px_vol_slope_{}'.format(interval_seconds)] = pv_cor_slope
    return df


def px_vol_corr_mean(df, interval_seconds=30, tick_interval_seconds=3):
    """
    在 interval_seconds 内, 量价相关性的均值
    """
    n_tick = interval_seconds // tick_interval_seconds
    pv_cor = ta.CORREL(df['LastPx'], df['TotalVolumeTrade'].diff(1), n_tick)
    pv_cor_mean = ta.SMA(pv_cor, n_tick)
    df['cor_px_vol_mean_{}'.format(interval_seconds)] = pv_cor_mean
    return df


def total_buysell_order_spread_ret_corr(df, interval_seconds=30, tick_interval_seconds=3):
    """
    待李浩补充
    """
    n_tick = interval_seconds // tick_interval_seconds
    # 把return放大scale倍
    ret_worldquant101 = np.log(df['LastPx']).diff(1) * scale
    df['TotalBuySellOrderQtyMinus_ret_corr_{}'.format(interval_seconds)] = ta.CORREL(ret_worldquant101,
                                                                                     df['TotalBuySellOrderQtyMinus'], n_tick)
    return df


def total_buysell_order_spread_sma_ret_corr(df, interval_seconds=30, tick_interval_seconds=3):
    """
    待李浩补充
    """
    n_tick = interval_seconds // tick_interval_seconds
    # 把return放大scale倍
    ret_worldquant101 = np.log(df['LastPx']).diff(1) * scale
    df['TotalBuySellOrderQtyMinus_retratio1_corr_{}'.format(interval_seconds)] = ta.CORREL(
        ta.SMA(ret_worldquant101, n_tick), df['TotalBuySellOrderQtyMinus'], n_tick)
    return df


def total_buysell_order_spread_sma(df, interval_seconds=30, tick_interval_seconds=3):
    """
    待李浩补充
    """
    n_tick = interval_seconds // tick_interval_seconds
    df['TotalBuySellOrderQtyMinus_mean_{}'.format(interval_seconds)] = ta.SMA(
        df['TotalBuySellOrderQtyMinus'], n_tick)
    return df


def total_buysell_order_spread_std(df, interval_seconds=30, tick_interval_seconds=3):
    """
    待李浩补充
    """
    n_tick = interval_seconds // tick_interval_seconds
    df['TotalBuySellOrderQtyMinus_std_{}'.format(interval_seconds)] = ta.STDDEV(
        df['TotalBuySellOrderQtyMinus'], n_tick)
    return df


def total_buysell_order_spread_vol_corr(df, interval_seconds=30, tick_interval_seconds=3):
    """
    待李浩补充
    """
    n_tick = interval_seconds // tick_interval_seconds
    df['TotalBuySellOrderQtyMinus_vol_corr_{}'.format(interval_seconds)] = ta.CORREL(df['TotalVolumeTrade'],
                                                                                     df['TotalBuySellOrderQtyMinus'], n_tick)
    return df


def weighted_price(df, level=10):
    def cal_mid_price(arrLike):
        price_bid = arrLike['BuyPriceQueue']
        price_ask = arrLike['SellPriceQueue']
        mid_price = (price_bid[0] + price_ask[0]) / 2
        return mid_price

    mid_price = df.apply(cal_mid_price, axis=1)

    for i in range(0, level):
        df['Weighted_Price_level{}'.format(i + 1)] = df.apply(
            lambda x: (x['BuyOrderQtyQueue'][i] * x['BuyPriceQueue'][i] + x['SellOrderQtyQueue'][i] * x['SellPriceQueue'][i]) \
                      / (x['BuyOrderQtyQueue'][i] + x['SellOrderQtyQueue'][i]),
            axis=1)

    return df


def VOLR(df, beta1=0.551, beta2=0.778, beta3=0.699):
    def get_VOLR(arrLike, beta1, beta2, beta3):
        beta = [beta1, beta2, beta3]
        volr = 0
        for i in range(0, 3):
            volr += beta[i] * (arrLike['BuyPriceQueue'][i] - arrLike['SellPriceQueue'][i]) / \
                    (arrLike['BuyPriceQueue'][i] + arrLike['SellPriceQueue'][i])
        return volr
    df['VOLR'] = df.apply(get_VOLR, **{"beta1": beta1, "beta2": beta2, "beta3": beta3}, axis=1)
    return df


def OFI(df, wlist=[0.27935, 0.256246, 0.323978], blist=[0.28864, 0.011068, 0.082937], t=10):
    def self_sign(x):
        x[x >= 0] = 1
        x[x < 0] = 0
        return x

    df['bp1'] = df.apply(lambda x: x['BuyPriceQueue'][0], axis=1)
    df['bp2'] = df.apply(lambda x: x['BuyPriceQueue'][1], axis=1)
    df['bp3'] = df.apply(lambda x: x['BuyPriceQueue'][2], axis=1)

    df['ap1'] = df.apply(lambda x: x['SellPriceQueue'][0], axis=1)
    df['ap2'] = df.apply(lambda x: x['SellPriceQueue'][1], axis=1)
    df['ap3'] = df.apply(lambda x: x['SellPriceQueue'][2], axis=1)

    df['bv1'] = df.apply(lambda x: x['BuyOrderQtyQueue'][0], axis=1)
    df['bv2'] = df.apply(lambda x: x['BuyOrderQtyQueue'][1], axis=1)
    df['bv3'] = df.apply(lambda x: x['BuyOrderQtyQueue'][2], axis=1)

    df['av1'] = df.apply(lambda x: x['SellOrderQtyQueue'][0], axis=1)
    df['av2'] = df.apply(lambda x: x['SellOrderQtyQueue'][1], axis=1)
    df['av3'] = df.apply(lambda x: x['SellOrderQtyQueue'][2], axis=1)

    if len(df) < t + 1: #
        # 如果数据量较少，该部分因子直接设置为0
        for factor in ['OFI1', 'OFI2', 'OFI3', 'WOFI']:
            df[factor] = 0
    else:
        for n in range(3):
            df['OFI' + str(n + 1)] = 0  # 首先设置新的因子的列
            flag = str(n + 1)
            for i in range(t):
                df0 = df.shift(i)  # 对数据进行滑动
                df1 = df.shift(i + 1)  # 对数据进行滑动
                # OFI因子，只是一个符号，正负的符号，而不需要具体的值
                'BuyPriceQueue'
                df['OFI' + str(n + 1) + '_' + str(i)] = self_sign(
                    df0.loc[:, 'bp' + flag] - df1.loc[:, 'bp' + flag]) * df0.loc[:, 'bv' + flag] - \
                                                        self_sign(df1.loc[:, 'bp' + flag] - df0.loc[:,
                                                                                            'bp' + flag]) * df1.loc[:,
                                                                                                            'bv' + flag] - \
                                                        self_sign(df1.loc[:, 'ap' + flag] - df0.loc[:,
                                                                                            'ap' + flag]) * df0.loc[:,
                                                                                                            'av' + flag] + \
                                                        self_sign(df0.loc[:, 'ap' + flag] - df1.loc[:,
                                                                                            'ap' + flag]) * df1.loc[:,
                                                                                                            'av' + flag]
                df['OFI' + str(n + 1) + '_' + str(i)] = df['OFI' + str(n + 1) + '_' + str(i)].fillna(0)  # 填上空值
            df['OFI' + str(n + 1)] = df.iloc[:, -t:].sum(axis=1)  # 对后t行求合作为因子
            # 删除掉产生的中间因子列
            for i in df.columns[-t:]:
                del (df[i])
            # 对于产生的因子再进行tanh得到最终的因子（结果一般为0, +1, -1）
            df['OFI' + str(n + 1)] = np.tanh(wlist[n] * df['OFI' + str(n + 1)])
        # 使用计算得到的OFI的因子计算WOFI因子
        df['WOFI'] = blist[0] * df['OFI1'] + blist[1] * df['OFI2'] + blist[2] * df['OFI3']
    del df['bp1'], df['bp2'], df['bp3'], df['ap1'], df['ap2'], df['ap3']
    del df['bv1'], df['bv2'], df['bv3'], df['av1'], df['av2'], df['av3']
    return df

# 聪明钱因子的排序指标
def winner_money(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    s_tick = n_tick // 5

    ret = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (df['LastPx'] + 1e-10)
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(n_tick)
    df['winner_money_{}'.format(interval_seconds)] = (abs(ret) + 1e-10) / np.sqrt(abs(volume + 1e-10))
    return df


# 聪明钱因子的排序指标
def winner_money_all(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    s_tick = n_tick // 2

    ret = (df['LastPx'] - df['LastPx'].shift(n_tick) + 1e-10) / (df['LastPx'] + 1e-10)
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(n_tick)
    st = (abs(ret) + 1e-10) / np.sqrt(abs(volume + 1e-10))
    vwap = ta.SUM((volume * df['LastPx']), n_tick) / ta.SUM(volume, n_tick)

    def sort_max(s):
        return sum(sorted(s, reverse=True)[:s_tick])

    df['winner_money_all_{}'.format(interval_seconds)] = (ta.SUM(vwap.rolling(n_tick).apply(sort_max),
                                                                 n_tick * 2) + 1e-10) / (
                                                                     ta.SUM(vwap, n_tick * 2) + 1e-10)
    return df


# 量在价先
def front_running(df, interval_seconds=30, tick_interval_seconds=3):
    n_tick = interval_seconds // tick_interval_seconds
    s_tick = n_tick // 5
    sr = df['LastPx'] - ta.SMA(df['LastPx'], n_tick)
    volume = df['TotalVolumeTrade'] - df['TotalVolumeTrade'].shift(1)
    sv = volume - ta.SMA(volume, n_tick)
    df['front_running_{}'.format(interval_seconds)] = ta.CORREL(sv, sr, n_tick * 2)
    return df

