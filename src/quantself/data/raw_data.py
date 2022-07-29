import os
import numpy as np
import pandas as pd
import ray
from ..configs import default_config

class RawData:
    def __init__(self, config=default_config):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        self.tradingcodes = self.config.pop("tradingcodes", ['000001'])
        self.date_interval = self.config.pop("date_interval", ['20220101', '20220601'])

    def _load_data(self):
        # 此处使用ray的task模式
        # step1. 首先确认具体的文件路径（暂定一个标的 一天为一个parquet）

        @ray.remote
        def load_data_utils():
            return
        # step2. 每个文件路径，启动一个ray的task拉取数据

        return

    def process_data(self):
        # 数据的resample
        @ray.remote
        def process_data_utils():
            return
        return


class ProcessedData:
    def __init__(self, processed_data_list):
        return

    def calculate_features(self):

        @ray.remote
        def calculate_features_utils():
            return
        return

    def calculate_labels(self):

        @ray.remote
        def calculate_labels_utils():
            return
        return

class Input:
    def __init__(self, input_list):
        return

    def transform_data(self):
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
