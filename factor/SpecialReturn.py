# 导入包
import numpy as np
from scipy.io import loadmat
import h5py
import time
# 导入其他部分代码
# 欲导入部分的路径置入
import sys
sys.path.append("module_1/factor")
sys.path.append("module_1/utils")
from ParamSet import *
from FactorExpoCalculator import *
from CalcValidStockMatrix import *


# 计算残差收益率
# 输入: paramSet-全局参数集合, daily_info-本地日频数据, factorExpo-因子暴露矩阵, factorReturn-因子收益率矩阵
# 输出: specialReturn-残差收益率计算结果
def SpecialReturn(paramSet, daily_info, factorExpo, factorReturn):
    # 计时器启动
    print('特质收益率计算')
    timeStart = time.clock()

    # 个股收益率序列
    closeAdj = daily_info['close_adj'][0][0]
    stockReturn = np.diff(closeAdj) / closeAdj[:, :(closeAdj.shape[1]-1)]
    stockReturn = np.concatenate((np.full((closeAdj.shape[0], 1), np.nan), stockReturn), axis=1)
    stockReturn = stockReturn[:, paramSet.updateTBegin-1:]
    [stockNum, dayNum] = np.shape(stockReturn)

    # 初始化结果
    specialReturn = np.full([stockNum, dayNum], np.nan)

    # 特异性收益率计算
    for iDay in range(1, dayNum):
        # 因子暴露
        panelExpo = factorExpo[:, :, iDay-1]
        # 因子收益率
        panelFactorReturn = (factorReturn[:, iDay])[:, np.newaxis].copy()
        panelFactorReturn[np.isnan(panelFactorReturn)] = 0
        # 计算残差收益率
        specialReturn[:, [iDay]] = (stockReturn[:, iDay])[:, np.newaxis] - np.dot(panelExpo, panelFactorReturn)
    
    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart))

    # 返回结果
    return specialReturn


if __name__ == '__main__':
    daily_info = loadmat('module_1/data/stock_data.mat', variable_names=['daily_info'])['daily_info']
    factorExpo = h5py.File('module_1/data/factorExpo.mat')['factorExpo']
    factorExpo = np.array(factorExpo)
    factorReturn = loadmat('module_1/data/factorReturn.mat', variable_names=['factorReturn'])['factorReturn']
    paramSet = ParamSet()
    paramSet.CalcUpdateTBegin(daily_info)
    specialReturn = SpecialReturn(paramSet, daily_info, factorExpo, factorReturn)
    print('finished')