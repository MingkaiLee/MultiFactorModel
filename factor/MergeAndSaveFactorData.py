# 导入包
import numpy as np
import time
import gc
from scipy.io import loadmat
import h5py
import sys
sys.path.append('module_1/utils')
from ParamSet import *


# 将增量部分计算结果和本地历史结果拼接
# 输入: paramSet-全局参数集合, factorExpoAdd-因子暴露数据, factorReturnAdd-因子收益率, specialReturnAdd-特质收益率
# 输出: factorExpo-因子暴露数据, factorReturn-因子收益率, specialReturn-特质收益率
def MergeAndSaveFactorData(paramSet, factorExpoAdd, factorReturnAdd, specialReturnAdd):
    # 计时器启动
    print('拼接计算结果')
    timeStart = time.clock()

    if paramSet.updateTBegin == 1:
        # 初始化模式
        factorExpo = np.array(factorExpoAdd, dtype=np.float)
        factorReturn = factorReturnAdd
        specialReturn = specialReturnAdd
    else:
        # 拼接因子暴露数据
        factorExpo_old = np.load('module_1/result/factorExpo.npy')
        n_stocks_before = factorExpo_old.shape[0]
        n_days_before = factorExpo_old.shape[2]
        n_stocks_after = factorExpoAdd.shape[0]
        n_days_add = factorExpoAdd.shape[2]
        n_days_after = paramSet.updateTBegin - 1 + n_days_add
        # 创建新的保留结果的矩阵
        factorExpo = np.full([n_stocks_after, 41, n_days_after], np.nan)
        # 旧值迁移
        factorExpo[:n_stocks_before, :, :n_days_before] = factorExpo_old
        # 新值赋予
        factorExpo[:n_stocks_after, :, paramSet.updateTBegin-1:n_days_after] = factorExpoAdd[:, :, :]
        # 删除不用的变量
        del factorExpo_old, factorExpoAdd
        gc.collect()

        # 拼接因子收益率数据
        factorReturn_old = np.load('module_1/result/factorReturn.npy')
        firstValidCol = np.where(np.sum(~np.isnan(factorReturnAdd), axis=0))[0]
        # 创建新的保留结果的矩阵
        factorReturn = np.full([n_stocks_after, n_days_after], np.nan)
        # 旧值迁移
        factorReturn[:n_stocks_before, :n_days_before] = factorReturn_old
        # 新值赋予
        factorReturn[:, paramSet.updateTBegin+firstValidCol-1:n_days_after] = factorReturnAdd[:, firstValidCol:]
        # 删除不用的变量
        del factorReturn_old, factorReturnAdd
        gc.collect()

        # 拼接残差收益率
        specialReturn_old = np.load('module_!/result/specialReturn.npy')
        firstValidCol = np.where(np.sum(~np.isnan(specialReturnAdd), axis=0))[0]
        # 创建新的保留结果的矩阵
        specialReturn = np.full([n_stocks_after, n_days_after], np.nan)
        # 旧值迁移
        specialReturn[:n_stocks_before, :n_days_before] = specialReturn_old
        # 新值赋予
        specialReturn[:n_stocks_after, paramSet.updateTBegin+firstValidCol-1:n_days_after] = specialReturnAdd[:, firstValidCol:]
        # 删除不用的变量
        del specialReturn_old, specialReturnAdd
        gc.collect()
    
    # 将拼接结果存放到目标路径
    np.save(paramSet.Global.save_path+'factorExpo.npy', factorExpo)
    np.save(paramSet.Global.save_path+'factorReturn.npy', factorReturn)
    np.save(paramSet.Global.save_path+'specialReturn.npy', specialReturn)

    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd-timeStart))

    return factorExpo, factorReturn, specialReturn


if __name__ == '__main__':
    factorExpo = h5py.File('module_1/result/factorExpo.mat')['factorExpo']
    factorExpo = np.array(factorExpo)
    factorReturn = loadmat('module_1/result/factorReturn.mat', variable_names=['factorReturn'])['factorReturn']
    specialReturn = loadmat('module_1/result/specialReturn.mat', variable_names=['specialReturn'])['specialReturn']
    paramSet = ParamSet()
    paramSet.CalcUpdateTBegin(0)
    factorExpo, factorReturn, specialReturn = MergeAndSaveFactorData(paramSet, factorExpo, factorReturn, specialReturn)
    print('end')