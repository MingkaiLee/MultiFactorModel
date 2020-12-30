# 多因子风险模型第二步: 生成因子协方差和残差协方差估计
# 导入包
import numpy as np
import gc
# 导入其他部分代码
# 欲导入部分的路径置入
import sys
sys.path.append('module_1/utils')
sys.path.append('module_1/estimate')
from FactorCovNeweyWest import *


# 因子协方差估计、估计效果评估及结果保存
def FactorCovAndEval(paramSet, factorReturn):
    factorCovNW = FactorCovNeweyWest(paramSet, factorReturn)