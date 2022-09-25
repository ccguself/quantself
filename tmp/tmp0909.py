import os
import numpy as np
import pandas as pd


# 关于cls目标分布情况
"""关于label有哪些需要分析的方面
1. 整体信号个数随着阈值设定的不同
2. 关于信号是否需要采样，overlap产生的问题到底是什么？如何解决？
3. 是否，3s一预测的情况下，其实可以默认为有新的信息产生？因此此时做一次样本的切分是合理的？


"""

class EDALabel:
    def __init__(self) -> None:
        return
    
    def eda_label_hist(self, threshold=1):
        return 