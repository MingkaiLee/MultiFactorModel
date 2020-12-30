# 因子协方差的Newey-West调整函数
# 导入包
import numpy as np
import time
import sys
sys.path.append('module_1/utils')
from ParamSet import *


def FactorCovNeweyWest(paramSet, factorReturn):
    """
    因子协方差的Newey-West调整
    ### Parameters
    paramSet(ParamSet)-参数集合, factorReturn(ndarray)-因子收益率序列
    ### Returns
    factorCov(ndarray)-因子协方差估计
    """
    # 计时器启动
    print('因子协方差矩阵Newey-West调整')
    timeStart = time.clock()

    # 参数准备
    tBegin = paramSet.nw.tBegin
    timeWin = paramSet.nw.timeWindow
    halfLife = paramSet.nw.halfLife
    dayNumOfMonth = paramSet.nw.dayNumOfMonth
    d = paramSet.nw.d

    # 获取因子收益率序列
    [styleNum, dayNum] = np.shape(factorReturn)

    # 半衰期权重序列
    weightList = np.power(0.5, np.arange(timeWin, 0, -1) / halfLife)

    # 初始化结果
    factorCov = np.full([styleNum, styleNum, dayNum], np.nan)

    # 遍历每个截面
    for iDay in range(tBegin-1, dayNum):
        # 获取窗口期样本
        data = factorReturn[:, iDay-timeWin+1:iDay+1]

        if np.sum(~np.isnan(data)) < 5:
            # 有效数值不足时避免不必要的计算
            factorCov[:, :, iDay] = np.nan
        
        else:
        
            # 计算因子协方差
            weight = weightList / np.sum(weightList)
            data = data - np.nansum(data*weight, axis=1)[:, np.newaxis]
            fnw = np.dot(np.dot(data, np.diag(weight)), data.T)

            # 考虑自相关项
            for q in range(0, d):
                k = 1 - (q+1) / (d+1)
                weight = weightList[q+1:] / np.sum(weightList[q+1:])
                dataLeft = np.dot(np.dot(data[:, q+1:], np.diag(weight)), data[:, :-(q+1)].T)
                dataRight = np.dot(np.dot(data[:, :-(q+1)], np.diag(weight)), data[:, q+1:].T)
                fnw = fnw + k * (dataLeft + dataRight)
                
            # 保存结果
            factorCov[:, :, iDay] = dayNumOfMonth * fnw
    
    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart))

    # 返回结果
    return factorCov


if __name__ == '__main__':
    factorReturn = np.load('module_1/result/factorReturn.npy')
    paramSet = ParamSet()
    x = FactorCovNeweyWest(paramSet, factorReturn)
    print('end')