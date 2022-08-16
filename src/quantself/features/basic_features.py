import os
import numpy as np
import pandas as pd
from ..calculators.factor import px_change_ratio

class BasicFeatures:
    def __init__(self, tick_interval=3):
        self.tick_interval = tick_interval
        return

    def calculate_features(self, processed_df):
        processed_df["PxChange_90"] = px_change_ratio(processed_df, 90, self.tick_interval)
        return