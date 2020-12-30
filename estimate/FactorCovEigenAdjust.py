# 因子协方差特征值调整
# 导入包
import numpy as np
import time
import sys
sys.path.append('module_1/utils')
from ParamSet import *


def factorCovEigenAdj(paramSet, factorCovNW):
    """
    因子协方差的特征值调整(效率较低, 运行增量部分, 并和历史结果拼接)
    ### Parameters
    paramSet(ParamSet)-参数集合, factorCovNW(ndarray)-因子收益率NW调整的结果
    ### Returns
    factorCovEigenAdj(ndarray)-特征值调整结果
    """
    print('因子协方差矩阵特征值调整')
    timeStart = time.clock()
    
