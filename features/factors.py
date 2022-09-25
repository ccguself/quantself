import os
import numpy as np
import pandas as pd
from features.factor_factory_old.price_volume_factors import *


class FactorsCalculator:
    def __init__(self) -> None:
        self.columns_drop = ["PreClosePx", "LastPx", 'OfferPx_0', 'OfferPx_1',
                    'OfferPx_2', 'OfferPx_3',
                    'OfferPx_4', 'OfferPx_5', 'OfferPx_6', 'OfferPx_7',
                    'OfferPx_8','OfferPx_9', 'OfferVol_0', 'OfferVol_1',
                    'OfferVol_2', 'OfferVol_3', 'OfferVol_4', 'OfferVol_5',
                    'OfferVol_6', 'OfferVol_7', 'OfferVol_8','OfferVol_9',
                    'BidPx_0', 'BidPx_1', 'BidPx_2', 'BidPx_3', 'BidPx_4',
                    'BidPx_5', 'BidPx_6', 'BidPx_7', 'BidPx_8', 'BidPx_9',
                    'BidVol_0','BidVol_1', 'BidVol_2', 'BidVol_3',
                    'BidVol_4', 'BidVol_5', 'BidVol_6','BidVol_7',
                    'BidVol_8', 'BidVol_9', 'Volume', 'IOPV', 'MidPx']
        pass

    def calculate_factors(self, df):

        # LastPx涨跌幅 - 近30s
        df.loc[:, "PxChange_30s"] = px_change_ratio(df, interval_seconds=30, tick_interval_seconds=3)

        # 剔除原始列
        df.drop(self.columns_drop, axis=1, inplace=True)
        return df