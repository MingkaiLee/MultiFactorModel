# 导入包
import numpy as np
from numpy.lib.twodim_base import diag
import pandas as pd
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
from SpecialReturn import *


# 计算因子收益率，将风格因子、行业因子、国家因子暴露汇总
# 输入: paramSet-全局参数, daily_info-本地日频数据, styleExpo-风格因子暴露集合, indusExpo-行业因子暴露矩阵, validStockMatrix-个股有效状态标记
# 输出: factorReturn-因子收益率计算结果, factorExpo-因子暴露汇总结果
def FactorReturn(paramSet, daily_info, styleExpo, indusExpo, validStockMatrix):
    # 计时器启动
    print('因子收益率计算')
    timeStart = time.clock()

    # 截断数据
    validStockMatrix = validStockMatrix[:, paramSet.updateTBegin-1:]
    indusExpo = np.array(indusExpo[:, paramSet.updateTBegin-1:], dtype=np.double)
    indusExpo = indusExpo * validStockMatrix

    # 各类因子数目
    styleNum = paramSet.Global.styleFactorNum
    indusNum = paramSet.Global.indusFactorNum
    countryNum = paramSet.Global.countryFactorNum
    factorNum = styleNum + indusNum + countryNum

    # 个股的收益率序列
    closeAdj = daily_info['close_adj'][0][0]
    stockReturn = np.diff(closeAdj) / closeAdj[:, :(closeAdj.shape[1]-1)]
    stockReturn = np.concatenate((np.full((closeAdj.shape[0], 1), np.nan), stockReturn), axis=1)
    stockReturn = stockReturn[:, paramSet.updateTBegin-1:] * validStockMatrix
    [stockNum, dayNum] = np.shape(stockReturn)

    # 权重数据
    close = daily_info['close'][0][0]
    floatAShares = daily_info['float_a_shares'][0][0]
    weightCap = close[:, paramSet.updateTBegin-1:] * floatAShares[:, paramSet.updateTBegin-1:]
    threshold = np.apply_along_axis(lambda x: prctile_matlab(x, paramSet.Global.mktBlockRatio), axis=0, arr=weightCap)
    threshold = threshold[np.newaxis, :]
    weightCap = np.minimum(weightCap, threshold)
    weightCap[np.isnan(weightCap)] = 0

    # 初始化结果
    factorReturn = np.full([factorNum, dayNum], np.nan)
    factorExpo = np.full([stockNum, factorNum, dayNum], np.nan)

    # 遍历每个截面
    for iDay in range(0, dayNum):
        # 获取截面上的风格因子暴露
        panelStyleExpo = np.concatenate([item[:, [iDay]] for item in styleExpo.values()], axis=1)
        
        # 获取行业因子哑变量矩阵
        panelIndusExpo = indusExpo[:, [iDay]]
        # 分前期29行业与后期30行业的情况
        if np.nanmax(panelIndusExpo) < 30:
            panelIndusExpo[np.isnan(panelIndusExpo)] = indusNum + 1
            panelIndusExpo = pd.get_dummies(panelIndusExpo.flatten())
            panelIndusExpo.insert(29, 30, 0)
            panelIndusExpo = np.array(panelIndusExpo)
        else:
            panelIndusExpo[np.isnan(panelIndusExpo)] = indusNum + 1
            panelIndusExpo = np.array(pd.get_dummies(panelIndusExpo.flatten()))
        panelIndusExpo = panelIndusExpo[:, :indusNum]

        # 保存因子暴露结果(面板数据)
        factorExpo[:, :, iDay] = np.concatenate([panelStyleExpo, panelIndusExpo, np.ones([stockNum, 1])], axis=1)

        # 最新一天的因子暴露已经无法计算收益率
        if iDay == dayNum-1:
            continue

        # 获取截面股票收益率(用T日因子对T+1日收益率回归)
        panelReturn = stockReturn[:, [iDay+1]]

        # 获取有完整数据的个股
        lineNoNan = ~np.isnan(np.sum(np.concatenate([panelReturn, panelStyleExpo, panelIndusExpo], axis=1), axis=1))
        if np.sum(lineNoNan) == 0:
            continue
        
        # 获取有效的回归自变量、因变量、权重
        y = panelReturn[lineNoNan, :]
        x = np.concatenate([panelStyleExpo, panelIndusExpo], axis=1)[lineNoNan, :]
        w = (weightCap[lineNoNan, iDay])[:, np.newaxis]

        # 回归系数估计(注意需要剔除全零列, 因为行业个数出现过变更)
        beta = np.full((x.shape[1], 1), np.nan)
        validCol = (np.sum(x!=0, axis=0) > 0)
        x = x[:, validCol]
        beta[validCol, :] = np.dot(np.linalg.inv(np.dot(np.dot(x.T, diag(w.flatten())), x)), np.dot(np.dot(x.T, diag(w.flatten())), y))

        # 风格因子收益率直接取回归结果
        styleReturn = beta[:styleNum]

        # 国家因子收益率是行业因子收益率的市值加权
        indusWeight = np.dot(w.T, panelIndusExpo[lineNoNan, :])
        indusReturn = beta[styleNum:indusNum+styleNum]
        countryReturn = np.nansum(indusReturn*indusWeight.T) / np.nansum((~np.isnan(indusReturn))*indusWeight.T)

        # 行业因子收益率是在原来基础上减去国家因子收益率
        indusReturn = indusReturn - countryReturn

        # 结果赋值, 注意这里存储的下标是T+1日, 也即基于T日因子暴露和T+1日收益率
        # 得到的回归结果是存在T+1日, 这样保证没有引入未来信息
        factorReturn[:, [iDay+1]] = np.concatenate([styleReturn, indusReturn, countryReturn.reshape((1, 1))], axis=0)

    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart))

    # 返回结果
    return factorReturn, factorExpo


if __name__ == '__main__':
    basic_info = loadmat('module_1/data/stock_data.mat', variable_names=['basic_info'])['basic_info']
    daily_info = loadmat('module_1/data/stock_data.mat', variable_names=['daily_info'])['daily_info']
    styleExpo = loadmat('module_1/data/styleExpo_std.mat', variable_names=['styleExpo'])['styleExpo']
    styleExpo = {'size': styleExpo['size'][0][0], 'beta': styleExpo['beta'][0][0], 'momentum': styleExpo['momentum'][0][0], 'residVola': styleExpo['residVola'][0][0],
    'nonLinear': styleExpo['nonLinear'][0][0], 'booktoPrice': styleExpo['booktoPrice'][0][0], 'liquidity': styleExpo['liquidity'][0][0], 
    'earningYield': styleExpo['earningYield'][0][0], 'growth': styleExpo['growth'][0][0], 'leverage': styleExpo['leverage'][0][0]}
    indusExpo = daily_info['cs_indus_code'][0][0]
    paramSet = ParamSet()
    paramSet.CalcUpdateTBegin(daily_info)
    validStockMatrix = CalcValidStockMatrix(basic_info, daily_info)
    factorReturn, factorExpo = FactorReturn(paramSet, daily_info, styleExpo, indusExpo, validStockMatrix)
    # 由于读取问题, 在此调试SpecialReturn函数
    specialReturn = SpecialReturn(paramSet, daily_info, factorExpo, factorReturn)
    print('Finished.')
