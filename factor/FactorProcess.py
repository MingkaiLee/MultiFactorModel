# 导入包
import numpy as np
from scipy.io import loadmat
import time
# 导入其他部分代码
# 欲导入部分的路径置入
import sys
sys.path.append("module_1/factor")
sys.path.append("module_1/utils")
from ParamSet import *
from FactorExpoCalculator import *
from CalcValidStockMatrix import *


# 风格因子暴露正交化、标准化
# 输入: paramSet-全局参数集合, daily_info-本地日频数据, styleFactor-10大类风格因子暴露字典, indusExpo-行业因子暴露矩阵, validStockMatrix-个股有效状态标记
# 输出: None
def FactorProcess(paramSet, daily_info, styleExpo, indusExpo, validStockMatrix):
    # 计时器启动
    print('因子暴露正交化、标准化')
    timeStart = time.clock()

    # 更新起始位置索引
    startLoc = paramSet.updateTBegin

    # 计算流通市值
    floatCap = daily_info['close'][0][0][:, startLoc-1:] * daily_info['float_a_shares'][0][0][:, startLoc-1:]

    # 对流通市值进行修正
    # 按分位数计算阈值
    threshold = np.apply_along_axis(lambda x: prctile_matlab(x, paramSet.Global.mktBlockRatio), axis=0, arr=floatCap)
    threshold = threshold[np.newaxis, :]
    # 修正过大市值
    floatCap = np.minimum(floatCap, threshold)

    # 将输入数据按照更新起始点截断
    for key, value in styleExpo.items():
        styleExpo[key] = value[:, startLoc-1:]
    indusExpo = indusExpo[:, startLoc-1:]
    validStockMatrix = validStockMatrix[:, startLoc-1:]

    # 因子标准化处理
    styleExpo['size'] = Standardize(paramSet, styleExpo['size'], floatCap, indusExpo, validStockMatrix)
    styleExpo['beta'] = Standardize(paramSet, styleExpo['beta'], floatCap, indusExpo, validStockMatrix)
    styleExpo['momentum'] = Standardize(paramSet, styleExpo['momentum'], floatCap, indusExpo, validStockMatrix)
    styleExpo['nonLinear'] = Standardize(paramSet, styleExpo['nonLinear'], floatCap, indusExpo, validStockMatrix)
    styleExpo['booktoPrice'] = Standardize(paramSet, styleExpo['booktoPrice'], floatCap, indusExpo, validStockMatrix)
    styleExpo['earningYield'] = Standardize(paramSet, styleExpo['earningYield'], floatCap, indusExpo, validStockMatrix)
    styleExpo['growth'] = Standardize(paramSet, styleExpo['growth'], floatCap, indusExpo, validStockMatrix)
    styleExpo['leverage'] = Standardize(paramSet, styleExpo['leverage'], floatCap, indusExpo, validStockMatrix)

    # 波动率因子对规模和Beta因子正交化,然后进行标准化
    baseFactor = [styleExpo['size'].copy(), styleExpo['beta'].copy()]
    styleExpo['residVola'] = OrthogonalizeM(styleExpo['residVola'], baseFactor, floatCap, validStockMatrix)
    styleExpo['residVola'] = Standardize(paramSet, styleExpo['residVola'], floatCap, indusExpo, validStockMatrix)

    # 流动性因子对规模因子正交化，然后进行标准化
    baseFactor = styleExpo['size'].copy()
    styleExpo['liquidity'] = Orthogonalize(styleExpo['liquidity'], baseFactor, floatCap, validStockMatrix)
    styleExpo['liquidity'] = Standardize(paramSet, styleExpo['liquidity'], floatCap, indusExpo, validStockMatrix)

    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart))

    return None


if __name__ == '__main__':
    basic_info = loadmat('module_1/data/stock_data.mat', variable_names=['basic_info'])['basic_info']
    daily_info = loadmat('module_1/data/stock_data.mat', variable_names=['daily_info'])['daily_info']
    styleExpo = loadmat('module_1/data/styleExpo.mat', variable_names=['styleExpo'])['styleExpo']
    styleExpo = {'size': styleExpo['size'][0][0], 'beta': styleExpo['beta'][0][0], 'momentum': styleExpo['momentum'][0][0], 'residVola': styleExpo['residVola'][0][0],
    'nonLinear': styleExpo['nonLinear'][0][0], 'booktoPrice': styleExpo['booktoPrice'][0][0], 'liquidity': styleExpo['liquidity'][0][0], 
    'earningYield': styleExpo['earningYield'][0][0], 'growth': styleExpo['growth'][0][0], 'leverage': styleExpo['leverage'][0][0]}

    paramSet = ParamSet()
    paramSet.CalcUpdateTBegin(daily_info)
    validStockMatrix = CalcValidStockMatrix(basic_info, daily_info)
    FactorProcess(paramSet, daily_info, styleExpo, daily_info['cs_indus_code'][0][0], validStockMatrix)
    print('end')
