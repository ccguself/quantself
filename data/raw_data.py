import os
import numpy as np
import pandas as pd
import datetime 
import ray
from configs import default_config, db_path
from features.factors import FactorsCalculator
from representation.labels import LabelsCalculator
from utils import trans_ExTime

class RawSnapshot:
    def __init__(self, config=default_config):
        self.config = config
        self._validate_config()
        self._get_file_path()
        self._load_data()

    def _validate_config(self):
        self.tradingcodes = self.config.pop("tradingcodes", ['510050.SH', '510300.SH'])
        self.date_interval = self.config.pop("date_interval", ['20220825'])

    def _get_file_path(self):
        self.file_paths = []
        self.tradingcode_list = []
        for tradingcode in self.tradingcodes:
            for date in self.date_interval:
                file = date + '.parquet'
                path = os.path.join(db_path, tradingcode, file)
                self.file_paths.append(path)
                self.tradingcode_list.append(tradingcode)

    def _load_data(self):
        columns_drop = ["OpenPx", "HighPx", "LowPx",
                        "NumTrades", "Turnover", "Status", "Ytm", "HighLimitPx",
                        "LowLimitPx", "SecPrefix", "Syl1", "Syl2", "SD2", "PreIOPV"]
                        
        # 此处使用ray的task模式
        @ray.remote
        def load_data_utils(path):
            raw_data = pd.read_parquet(path)
            raw_data.drop(columns_drop, axis=1, inplace=True)
            return raw_data
        # step2. 每个文件路径，启动一个ray的task拉取数据
        self.raw_data_list = ray.get([load_data_utils.remote(path) for path in self.file_paths])

    def process_data(self):
        # 数据的resample
        @ray.remote
        def process_data_utils(raw_df, tradingcode):
            """ 
            step1. 数据格式的预处理
                0. 只保留连续竞价时间的数据
                1. TradeDate：转换为datetime的数据类型
                2. ExTime：转换为datetime的数据类型
                3. 价格还原 / 1000
            
            step2. 降采样 + 消除降采样后产生的午休时间数据
            """
            # step1.
            if "SH" in tradingcode:
                raw_df = raw_df.loc[
                    ((raw_df["ExTime"] >= 93000000) & (raw_df["ExTime"] <= 113000000))
                    | ((raw_df["ExTime"] >= 130000000) & (raw_df["ExTime"] <= 150000000))
                    ]
            elif "SZ" in tradingcode:
                raw_df = raw_df.loc[
                    ((raw_df["ExTime"] >= 93000000) & (raw_df["ExTime"] <= 113000000))
                    | ((raw_df["ExTime"] >= 130000000) & (raw_df["ExTime"] <= 145700000))
                    ]

            raw_df["TradeDate"] = raw_df["TradeDate"].apply(lambda x:datetime.datetime.strptime(str(x), "%Y%m%d").date())
            raw_df["ExTime"] = raw_df["ExTime"].apply(trans_ExTime)
            raw_df["DateTime"] = raw_df.apply(lambda x:datetime.datetime.combine(x['TradeDate'], x['ExTime']), axis=1)
            columns_divide = ['LastPx', 'OfferPx_0', 'OfferPx_1', 'OfferPx_2', 'OfferPx_3',
                                'OfferPx_4', 'OfferPx_5', 'OfferPx_6', 'OfferPx_7', 'OfferPx_8',
                                'OfferPx_9', 'BidPx_0', 'BidPx_1', 'BidPx_2', 'BidPx_3',
                                'BidPx_4', 'BidPx_5', 'BidPx_6', 'BidPx_7', 'BidPx_8',
                                 'BidPx_9', 'WAvgBidPri', 'WAvgOfferPri', 'IOPV']
            raw_df.loc[:, columns_divide] = raw_df.loc[:, columns_divide] / 10000

            # step2. 
            # 对于resample的准备
            # 1. 保留和生成需要resample的列
            # 2. 各列resample的方法
            columns_to_keep = ['DateTime', "TradeDate", "ExTime", "PreClosePx", "LastPx", 
                                'OfferPx_0', 'OfferPx_1', 'OfferPx_2', 'OfferPx_3',
                                'OfferPx_4', 'OfferPx_5', 'OfferPx_6', 'OfferPx_7',
                                'OfferPx_8','OfferPx_9', 'OfferVol_0', 'OfferVol_1',
                                'OfferVol_2', 'OfferVol_3', 'OfferVol_4', 'OfferVol_5',
                                'OfferVol_6', 'OfferVol_7', 'OfferVol_8','OfferVol_9',
                                'BidPx_0', 'BidPx_1', 'BidPx_2', 'BidPx_3', 'BidPx_4',
                                'BidPx_5', 'BidPx_6', 'BidPx_7', 'BidPx_8', 'BidPx_9',
                                'BidVol_0','BidVol_1', 'BidVol_2', 'BidVol_3',
                                'BidVol_4', 'BidVol_5', 'BidVol_6','BidVol_7',
                                'BidVol_8', 'BidVol_9', 'Volume', 'IOPV']

            # raw_df = raw_df.loc[raw_df["DateTime"]]
            # raw_df["IntervalOpenPx"] = raw_df["LastPx"]
            # raw_df["IntervalClosePx"] = raw_df["LastPx"]
            # raw_df["IntervalHighPx"] = raw_df["LastPx"]
            # raw_df["IntervalLowPx"] = raw_df["LastPx"]
            # raw_df["Twap"] = raw_df["LastPx"]
            raw_df["Volume"] = raw_df["Volume"].diff(1)
            raw_df = raw_df.loc[:, columns_to_keep]

            # 添加一些基础列
            raw_df.loc[:, "MidPx"] = (raw_df.loc[:, "OfferPx_0"] + raw_df.loc[:, "BidPx_0"]) / 2

            return raw_df
        self.processed_data_list = ray.get([process_data_utils.remote(raw_df, tradingcode) for raw_df, tradingcode in zip(self.raw_data_list, self.tradingcode_list)])


class ProcessedSnapshot:
    def __init__(self):
        return

    # 此处各个task公用一个因子计算器（FactorCalculator）
    
    def calculate_features(self, processed_data_list):
        public_calculator = FactorsCalculator()
        ray_calculator = ray.put(public_calculator)
        @ray.remote
        def calculate_features_utils(factors_calculator, processed_data):
            factors_data = factors_calculator.calculate_factors(processed_data)
            return factors_data
        self.factors_data_list = ray.get([calculate_features_utils.remote(ray_calculator, processed_data) for processed_data in processed_data_list])

    def calculate_labels(self, processed_data_list):
        public_calculator = LabelsCalculator()
        ray_calculator = ray.put(public_calculator)
        @ray.remote
        def calculate_labels_utils(label_calculator, processed_data):
            labels_data = label_calculator.calculate_labels(processed_data)
            return labels_data
        self.labels_data_list = ray.get([calculate_labels_utils.remote(ray_calculator, processed_data) for processed_data in processed_data_list])

class Input:
    def __init__(self):

        return

    def transform_data(self, factors_data_list, labels_data_list):
        # 此处需要考虑是否严格不使用未来信息
        # 如果严格不使用，则先按照单个dataframe获得具体的规范化值，例如归一化，缺失值处理，波动率计算等
        # 暂定为严格不使用未来信息
        @ray.remote
        def get_calculator_utils():
            return
        return

    def cross_validate(self):
        # 此处参照word2vec的取数构造逻辑，直接生成训练、验证、测试数据
        return
