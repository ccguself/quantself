""" 需要引入的配置项
1. 取数相关
    a. 涉及标的
    b. 日期区间
2. 打标相关
    a. 观测窗口长度（采样窗口长度 为窗口最小变动单位） - 训练窗口
    b. 观测窗口构建是否重叠
    c. 验证窗口长度（采样窗口长度 为窗口最小变动单位） - 打标窗口
3. 采样相关
    a. 采样频率
    b.
4. 模型训练相关
    a. 验证集构建方式：cross_validate
    b.
    c.

"""
import os
project_path = 'd:\\Books'
db_path = 'e:\\Books\\Data'


default_config = {}