import numpy as np
import talib as ta
import pandas as pd
from .utils import ts_rank, scale
from pyfinance.ols import PandasRollingOLS


def order_imbalance(df: pd.DataFrame, level: int = 1):
    """
    第i档买卖盘数量的不均衡,
    source: https://mp.weixin.qq.com/s?__biz=MzkxMDE2NDc2Mw==&mid=2247484177&amp;idx=1&amp;sn=8e9f82e9b8e3ed4d5d15f774555e80d1&source=41#wechat_redirect
    """
    level = level - 1
    current_buy_vol = df["BuyOrderQtyQueue"].apply(lambda x: x[level])
    current_sell_vol = df["SellOrderQtyQueue"].apply(lambda x: x[level])
    order_imbalance = (current_buy_vol - current_sell_vol)
    return order_imbalance.values

def order_imbalance_ratio(df: pd.DataFrame, level: int = 1):
    """
    第i档买卖盘数量的不均衡比例
    source: https://mp.weixin.qq.com/s?__biz=MzkxMDE2NDc2Mw==&mid=2247484177&amp;idx=1&amp;sn=8e9f82e9b8e3ed4d5d15f774555e80d1&source=41#wechat_redirect
    """
    level = level - 1
    current_buy_vol = df["BuyOrderQtyQueue"].apply(lambda x: x[level])
    current_sell_vol = df["SellOrderQtyQueue"].apply(lambda x: x[level])
    order_imbalance_ratio = (current_buy_vol - current_sell_vol) / (current_buy_vol + current_sell_vol)
    return order_imbalance_ratio.values

def speed_of_size(df: pd.DataFrame, level: int=1, type:str ='buy', interval_seconds: int=30, tick_interval_seconds: int=3):
    """
    speed_of_size
    """
    n_tick = interval_seconds // tick_interval_seconds
    bid_vol = df["BuyOrderQtyQueue"].apply(lambda x: x[level])
    ask_vol = df["SellOrderQtyQueue"].apply(lambda x: x[level])
    if type == 'buy':
        speed = bid_vol.diff(n_tick) / interval_seconds * scale
    else:
        speed = ask_vol.diff(n_tick) / interval_seconds * scale

    return speed


def buy_sell_order_qty_ratio(df, level=10):
    """
    实时10档盘口买卖盘总委托比
    """
    for i in range(0, level):
        df['OrderQtyBuySellRatio{}'.format(i + 1)] = df.apply(
            lambda x: np.log((x['BuyOrderQtyQueue'][i] + 1e-10) /
                             (x['SellOrderQtyQueue'][i] + 1e-10)),
            axis=1)
    return df


def total_buy_sell_qty_ratio(df):
    """
    实时10档盘口总委托比(log)
    """
    df['TotalBuySellOrderQtyRatio'] = df.apply(
        lambda x: np.log((np.sum(x['BuyOrderQtyQueue']) + 1e-10) / (np.sum(x['SellOrderQtyQueue']) + 1e-10)), axis=1)
    return df


def total_buy_sell_qty_ratio_change(df, interval_seconds=30, tick_interval_seconds=3):
    """
    在 interval_seconds 内, 实时10档盘口总委托比(log)的变化
    """
    n_tick = interval_seconds // tick_interval_seconds
    return df['TotalBuySellOrderQtyRatio'].diff(n_tick)



def relative_volume(df, interval_seconds=30, tick_interval_seconds=3):
    """
    近N秒 成交量 / 当日 N秒 成交量均值
    """
    df_index = pd.RangeIndex(0, len(df))
    n_tick = interval_seconds // tick_interval_seconds
    return df['TotalVolumeTrade'].diff(n_tick) / (
        df['TotalVolumeTrade'] * n_tick / (df_index + 1) + 1e-10)


def total_buysell_orderqty_ratio(df):
    '''
    total_buysell_orderqty_ratio  log实时10档盘口总委托比
    '''
    df['TotalBuySellOrderQtyRatio'] = df.apply(lambda x: np.log((np.sum(x['BuyOrderQtyQueue'])+ 1e-10) / (np.sum(x['SellOrderQtyQueue'])+ 1e-10)), axis=1)
    return df




def vol_ratio_mean_ratio(df, interval_seconds=30, tick_interval_seconds=3):
    '''
    volratio_mean_ratio 成交量比率因子
    '''
    n_tick = interval_seconds // tick_interval_seconds
    df['volratio_mean_ratio_{}'.format(interval_seconds)] = ta.SMA((df['TotalVolumeTrade']+ 1e-10) / (ta.SMA(df['TotalVolumeTrade'], n_tick * 2)+ 1e-10), n_tick)
    return df



def volume_raito_ret_corr(df, interval_seconds=30, tick_interval_seconds=3):
    '''
    volume_raito_ret_corr 成交量的一些衍生因子
    '''
    n_tick = interval_seconds // tick_interval_seconds
    df['volume_raito_ret_corr_{}'.format(interval_seconds)] = ta.CORREL(df['LastPx'] / df['LastPx'].shift(1),
                                                              df['TotalVolumeTrade'] / df['TotalVolumeTrade'].shift(1),
                                                              n_tick)

    return df


def volume_raito_ret_corr_volume(df, interval_seconds=30, tick_interval_seconds=3):
    '''
    volume_raito_ret_corr_volume 成交量的一些衍生因子
    '''
    n_tick = interval_seconds // tick_interval_seconds
    df['volume_raito_ret_corr_{}_volume'.format(interval_seconds)] = ta.CORREL(df['LastPx'] / df['LastPx'].shift(1),
                                                                     df['TotalVolumeTrade'] / df[
                                                                         'TotalVolumeTrade'].shift(1), n_tick) * df[
                                                               'TotalVolumeTrade']
    return df


def volume_raito_ret_corr_volumeratio(df, interval_seconds=30, tick_interval_seconds=3):
    '''
    volume_raito_ret_corr_volumeratio 成交量的一些衍生因子
    '''
    n_tick = interval_seconds // tick_interval_seconds
    temp = df['TotalVolumeTrade'] / ta.SMA(df['TotalVolumeTrade'], n_tick)
    df['volume_raito_ret_corr_{}_volumeratio'.format(interval_seconds)] = ta.CORREL(df['LastPx'] / df['LastPx'].shift(1),
                                                                          df['TotalVolumeTrade'] / df[
                                                                              'TotalVolumeTrade'].shift(1), n_tick) * temp

    return df


def volume_sma_ratio_ret_corr(df, interval_seconds=30, tick_interval_seconds=3):
    '''
    volume_sma_ratio_ret_corr 成交量的一些衍生因子
    '''
    n_tick = interval_seconds // tick_interval_seconds
    temp = df['TotalVolumeTrade'] / ta.SMA(df['TotalVolumeTrade'], n_tick)
    df['volume_sma_ratio_ret_corr_{}'.format(interval_seconds)] = ta.CORREL(df['LastPx'] / df['LastPx'].shift(1),
                                                                  temp, n_tick)
    return df


def volume_sma_ratio_ret_corr_closeopen(df, interval_seconds=30, tick_interval_seconds=3):
    """
    
    """
    n_tick = interval_seconds // tick_interval_seconds
    temp = df['TotalVolumeTrade'] / ta.SMA(df['TotalVolumeTrade'], n_tick)

    df['volume_sma_ratio_ret_corr_{}_closeopen'.format(interval_seconds)] = ta.CORREL(df['LastPx'] / df['LastPx'].shift(1),
                                                                            temp,
                                                                            n_tick) * abs(
        df['LastPx'] / df['OpenPx'] - 1)

    return df


def volume_sma_ratio_ret_corr_volume(df, interval_seconds=30, tick_interval_seconds=3):
    """
    
    """
    n_tick = interval_seconds // tick_interval_seconds
    temp = df['TotalVolumeTrade'] / ta.SMA(df['TotalVolumeTrade'], n_tick)
    df['volume_sma_ratio_ret_corr_{}_volume'.format(interval_seconds)] = ta.CORREL(df['LastPx'] / df['LastPx'].shift(1),
                                                                         temp, n_tick) * \
                                                               df['TotalVolumeTrade']

    return df


def volume_sma_ratio_ret_corr_volumeratio(df, interval_seconds=30, tick_interval_seconds=3):
    """
    
    """
    n_tick = interval_seconds // tick_interval_seconds
    temp = df['TotalVolumeTrade'] / ta.SMA(df['TotalVolumeTrade'], n_tick)
    df['volume_sma_ratio_ret_corr_{}_volumeratio'.format(interval_seconds)] = ta.CORREL(df['LastPx'] / df['LastPx'].shift(1),
                                                                              temp,
                                                                              n_tick) * temp
    return df
