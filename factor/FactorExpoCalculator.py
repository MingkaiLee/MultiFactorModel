### 因子暴露计算工具包
# 导入包
import numpy as np
from numpy.core.fromnumeric import mean, shape
from numpy.lib.twodim_base import diag
import pandas as pd
import time
from copy import deepcopy


## 计算辅助工具
# 计算列加权标准差
# 输入: vals-待求标准差矩阵, weights-权重矩阵
# 输出: stds-按weights进行列元素加权的标准差矩阵
def WeightedColStd(vals, weights, omitnan=False):
    """
    计算列加权标准差
    ### Parameters
    vals(ndarray)-待求标准差矩阵, weights(ndarray)-权重矩阵, omitnan(boolean)-是否忽略nan
    ### Returns
    stds(ndarray)-按weights进行列元素加权的标准差矩阵
    """
    variances = None
    # 仿写matlab的omitnan模式
    if omitnan:
        # 扩展权重矩阵
        weights_tiled = np.tile(weights, (vals.shape[0], 1))
        # 计算行元素加权值
        weighted_val = vals * weights_tiled
        # 获取行元素有效权值和
        weights_tiled_nan = deepcopy(weights_tiled)
        weights_tiled_nan[np.isnan(weighted_val)] = 0
        weights_sum = weights_tiled_nan.sum(axis=1).reshape((weights_tiled_nan.shape[0], 1))
        # 计算加权均值
        avgs = np.nansum(weighted_val, axis=1).reshape((vals.shape[0], 1)) / weights_sum
        
        # 计算方差矩阵行元素加权值
        weighted_var = np.power(vals-avgs, 2) * weights_tiled  
        # 计算加权方差
        variances = np.nansum(weighted_var, axis=1).reshape((vals.shape[0], 1)) / weights_sum

    # 普通模式,与matlab的all模式对应
    else:
        # 计算加权均值
        avgs = np.average(vals, axis=1, weights=weights).reshape((vals.shape[0], 1))

        # 计算加权方差
        variances = np.average(np.power(vals-avgs, 2), axis=1, weights=weights).reshape((avgs.shape[0], 1))

        # 返回加权方差的平方根
    return np.sqrt(variances)

# 根据半衰期计算移动加权矩阵
# 输入: dataMatrix-待加权矩阵, timeWin-数据时窗长度, halfLife-指数衰减的半衰期,不输入表示等权
# 输出: weightMatrix-加权矩阵
def CalcMovingWeightMatrix(dataMatrix, timeWin, halfLife=None):
    """
    根据半衰期计算移动加权矩阵
    ### Parameters
    dataMatrix(ndarray)-待加权矩阵, timeWin(int)-数据时窗长度, halfLife(int)-指数衰减的半衰期,不输入表示等权
    ### Returns
    输出: weightMatrix(ndarray)-加权矩阵
    """
    # 计算权重向量
    weight = None
    if halfLife:
        weight = np.power(1/2, np.arange(timeWin, 0, -1) / halfLife)
    else:
        weight = np.ones((1, timeWin))
    weight = weight.reshape((1, weight.shape[0]))

    # 生成一维向量
    zeroInterval = np.zeros((1, dataMatrix.shape[1]-timeWin+1))
    tempWghtMtr = np.tile(np.concatenate((weight, zeroInterval), axis=1), (1, dataMatrix.shape[1]-timeWin))
    tempWghtMtr = np.concatenate((tempWghtMtr, weight), axis=1)
    
    # 将向量重塑为矩阵
    weightMatrix = tempWghtMtr.reshape((dataMatrix.shape[1]-timeWin+1, dataMatrix.shape[1]))
    weightMatrix = weightMatrix.T

    # 归一化
    weightMatrix = weightMatrix / np.sum(weightMatrix, axis=0)

    return weightMatrix

# 计算matlab里的百分位函数
# 输入: x-1维数据, p-分位数
# 输出: y-对应百分位数
def prctile_matlab(x, p):
    """
    计算matlab里的百分位函数
    ### Parameters
    x(ndarray)-1维数据, p(float)-分位数
    ### Returns
    y(float)-对应百分位数
    """
    # 转为Series对象
    x = pd.Series(x)
    
    # X: Series
    # p: scalar
    x = x.sort_values().dropna().values
    n = len(x)

    # Form the vector of index values (numel(p) x 1)
    r = (p / 100) * n
    k = np.floor(r + 0.5).astype(int)  # K gives the index for the row just before r
    kp1 = k + 1  # K+1 gives the index for the row just after r
    r = r - k  # R is the ratio between the K and K+1 rows

    # Find indices that are out of the range 1 to n and cap them
    if k < 1 or np.isnan(k):
        k = 1
    kp1 = np.min([kp1, n])

    # Use simple linear interpolation for the valid percentages
    y = (0.5 + r) * x[kp1 - 1] + (0.5 - r) * x[k - 1]

    # Make sure that values we hit exactly are copied rather than interpolated
    exact = (r == -0.5)
    if exact:
        y = x[k - 1]

    # Make sure that identical values are copied rather than interpolated
    same = (x[k - 1] == x[kp1 - 1])
    if same:
        x = x[k - 1]  # expand x
        y = x
    return y

# 将因子暴露数据进行中位数去极值、缺失值补行业均值、标准化处理
# 输入: paramSet-全局参数集合, oldExpo-预处理前的因子暴露数据, floatCap-流通市值, indusFactorExpo-行业因子暴露矩阵, validStockMatrix-个股有效状态标记
# 输出: newExpo-标准化后的因子暴露数据
def Standardize(paramSet, oldExpo, floatCap, indusFactorExpo, validStockMatrix):
    """
    将因子暴露数据进行中位数去极值、缺失值补行业均值、标准化处理
    ### Parameters
    paramSet(ParamSet)-全局参数集合, oldExpo(ndarray)-预处理前的因子暴露数据, floatCap(ndarray)-流通市值, indusFactorExpo(ndarray)-行业因子暴露矩阵, 
    validStockMatrix(ndarray)-个股有效状态标记
    ### Returns
    newExpo(ndarray)-标准化后的因子暴露数据
    """
    # 初始化结果
    newExpo = oldExpo * validStockMatrix
    [stockNum, dayNum] = newExpo.shape

    ## 中位数去极值
    # Z-score去极值中的阈值(中位数的倍数)
    thZscore = paramSet.Global.thZscore

    # 将超出门限值的极端值设为门限值
    medianData = np.nanmedian(newExpo, axis=0)[np.newaxis, :]
    tempData = np.abs(newExpo-medianData)
    medianTempData = np.nanmedian(tempData, axis=0)[np.newaxis, :]
    thresholdHMatrix = np.tile(medianData + thZscore*medianTempData, [stockNum, 1])
    thresholdLMatrix = np.tile(medianData - thZscore*medianTempData, [stockNum, 1])
    newExpo[newExpo>thresholdHMatrix] = thresholdHMatrix[newExpo>thresholdHMatrix]
    newExpo[newExpo<thresholdLMatrix] = thresholdLMatrix[newExpo<thresholdLMatrix]
    
    ## 填充缺失值
    # 基于每个截面上行业内包含的有效数据计算行业均值
    indusMean = np.full([stockNum, dayNum], np.nan)
    for iDay in range(0, dayNum):
        for iIndus in range(1, paramSet.Global.indusFactorNum+1):
            # 获取当前截面上属于目标行业的个股集合
            includeStockIndex = (indusFactorExpo[:, iDay] == iIndus)
            # 计算目标行业中有多少只股票
            includeStockNum = np.sum(includeStockIndex)
            # 计算有效数据的均值
            tempMean = np.nanmean(newExpo[includeStockIndex, iDay])
            # 计算有效数据的个数
            validDataNum = np.sum(~np.isnan(newExpo[includeStockIndex, iDay]))
            # 只有当数据足够时才将缺失值填充为行业均值, 否则直接填充为0
            if validDataNum > paramSet.Global.thDataTooFew:
                indusMean[includeStockIndex, iDay] = tempMean
            else:
                indusMean[includeStockIndex, iDay] = 0

    # 填充缺失值
    nanIndex = np.isnan(newExpo)
    newExpo[nanIndex] = indusMean[nanIndex]

    # 再次置为无效数据(可能有些无效的股票也被填充行业均值了)
    newExpo = newExpo * validStockMatrix

    ## 中心化, 标准化
    # 根据参数设置决定是否对市值进行开根号处理
    if paramSet.Global.capSqrtFlag == 1:
        floatCap = np.sqrt(floatCap)
    
    # 将无效股票处的市值置为空值
    floatCap[np.isnan(newExpo)] = np.nan

    # 计算截面因子暴露均值(市值加权)
    meanExpo = np.nansum(newExpo*floatCap, axis=0) / np.nansum(floatCap, axis=0)
    
    # 中心化
    newExpo = newExpo - meanExpo[np.newaxis, :]
    
    # 标准化
    newExpo = newExpo / np.nanstd(newExpo, axis=0)[np.newaxis, :]
    
    # 返回结果
    return newExpo

# 将目标因子对基准因子(1个)按流通市值加权正交化
# 算法基本假设: targetFactor[] = resultFactor[] + beta*baseFactor + error, 其中resultFactor就是正交后的因子暴露, 与baseFactor完全正交, 互不相关
# 输入: targetFactor-待正交的因子暴露, baseFactor-正交的因子暴露矩阵, weight-加权最小二乘权重, validStockMatrix-可交易股票的标记矩阵
# 输出: resultFactor-正交余量, 作为新的因子暴露值
def Orthogonalize(targetFactor, baseFactor, weight, validStockMatrix):
    """
    将目标因子对基准因子(1个)按流通市值加权正交化

    算法基本假设: targetFactor[] = resultFactor[] + beta*baseFactor + error, 其中resultFactor就是正交后的因子暴露, 与baseFactor完全正交, 互不相关
    ### Parameters
    targetFactor(ndarray)-待正交的因子暴露, baseFactor(ndarray)-正交的因子暴露矩阵, weight(ndarray)-加权最小二乘权重, validStockMatrix(ndarray)-可交易股票的标记矩阵
    ### Returns
    resultFactor(ndarray)-正交余量, 作为新的因子暴露值
    """
    # 获取有效因子暴露
    targetFactor = targetFactor * validStockMatrix

    # 获取股票维度与日期维度
    [stockNum, dayNum] = targetFactor.shape

    # 初始化结果
    resultFactor = np.full([stockNum, dayNum], np.nan)

    # 遍历每个截面
    for iDay in range(0, dayNum):
        # 自变量, 因变量, 权重
        y = targetFactor[:, [iDay]]
        x = baseFactor[:, [iDay]]
        w = weight[:, [iDay]]
        # 保留三个值均有效的索引
        validStock = ~np.isnan(np.sum(np.concatenate([y, x, w], axis=1), axis=1))
        # 解决空valid变量问题
        if np.sum(validStock) == 0:
            ret = np.full((2, 1), np.nan)
        else:
            validY = y[validStock, :]
            validX = x[validStock, :]
            validW = w[validStock, :]
            # 直接基于解析解得到回归系数估计
            tmpX = np.concatenate([validX, np.ones([validX.shape[0], 1])], axis=1)
            tmpW = np.diag(validW.flatten())
            ret = np.dot(np.linalg.inv(np.dot(np.dot(tmpX.T, tmpW), tmpX)), np.dot(np.dot(tmpX.T, tmpW), validY))
        beta = ret[0, 0]
        alpha = ret[1, 0]
        resultFactor[:, [iDay]] = y - beta*x - alpha * np.ones([stockNum, 1])
    
    # 返回结果
    return resultFactor

# 将目标因子对基准因子(多个)按流通市值加权正交化
# 算法基本假设: targetFactor[] = resultFactor[] + beta*baseFactor + error, 其中resultFactor就是正交后的因子暴露, 与baseFactor完全正交, 互不相关
# 输入: targetFactor-待正交的因子暴露, baseFactor-正交的因子暴露矩阵的列表, weight-加权最小二乘权重, validStockMatrix-可交易股票的标记矩阵
# 输出: resultFactor-正交余量, 作为新的因子暴露值
def OrthogonalizeM(targetFactor, baseFactorList, weight, validStockMatrix):
    """
    将目标因子对基准因子(多个)按流通市值加权正交化

    算法基本假设: targetFactor[] = resultFactor[] + beta*baseFactor + error, 其中resultFactor就是正交后的因子暴露, 与baseFactor完全正交, 互不相关
    ### Parameters
    targetFactor(ndarray)-待正交的因子暴露, baseFactor(List)-正交的因子暴露矩阵的列表, weight(ndarray)-加权最小二乘权重, validStockMatrix(ndarray)-可交易股票的标记矩阵
    ### Returns
    resultFactor(ndarray)-正交余量, 作为新的因子暴露值
    """
    # 获取有效因子暴露
    targetFactor = targetFactor * validStockMatrix

    # 获取股票维度与日期维度
    [stockNum, dayNum] = targetFactor.shape

    # 初始化结果
    resultFactor = np.full([stockNum, dayNum], np.nan)

    # 遍历每个截面
    for iDay in range(0, dayNum):
        # 自变量, 因变量, 权重
        y = targetFactor[:, [iDay]]
        # x数据合成
        x = np.concatenate([item[:, [iDay]] for item in baseFactorList], axis=1)
        w = weight[:, [iDay]]
        # 保留三个值均有效的索引
        validStock = ~np.isnan(np.sum(np.concatenate([y, x, w], axis=1), axis=1))
        # 解决空valid变量问题
        if np.sum(validStock) == 0:
            ret = np.full((len(baseFactorList)+1, 1), np.nan)
        else:    
            validY = y[validStock, :]
            validX = x[validStock, :]
            validW = w[validStock, :]
            # 直接基于解析解得到回归系数估计
            tmpX = np.concatenate([validX, np.ones([validX.shape[0], 1])], axis=1)
            tmpW = np.diag(validW.flatten())
            ret = np.dot(np.linalg.inv(np.dot(np.dot(tmpX.T, tmpW), tmpX)), np.dot(np.dot(tmpX.T, tmpW), validY))
        beta = (ret[0:ret.shape[0]-1, 0])[:, np.newaxis]
        alpha = ret[ret.shape[0]-1, 0]
        resultFactor[:, [iDay]] = y - np.dot(x, beta) - alpha * np.ones([stockNum, 1])
    
    # 返回结果
    return resultFactor

# 将原始财报公布数据(口径为累计值)转换为TTM值
# 转换公式: 本期ttm = 本期报告 + 最新一次(不算本期)年报 - 上一年同期报告
# 输入: data-待转换数据
# 输出: ttmData-转换数据
def TransformTTM(data):
    """
    将原始财报公布数据(口径为累计值)转换为TTM值

    转换公式: 本期ttm = 本期报告 + 最新一次(不算本期)年报 - 上一年同期报告
    ### Parameters
    data(ndarray)-待转换数据
    ### Returns
    ttmData(ndarray)-转换数据
    """
    data = np.array(data)
    # 去年同期报告
    quarterYearBeforeData = np.hstack([np.full([data.shape[0], 4], np.nan), deepcopy(data[:, :-4])])
    
    # 去年年报(注意本地数据库第一个季报是1997年年报)
    lastYearData = deepcopy(data)
    maxCol = lastYearData.shape[1]
    lastYearData[:, range(1, maxCol, 4)] = data[:, range(0, maxCol-1, 4)]
    lastYearData[:, range(2, maxCol, 4)] = data[:, range(0, maxCol-2, 4)]
    lastYearData[:, range(3, maxCol, 4)] = data[:, range(0, maxCol-3, 4)]
    lastYearData[:, range(4, maxCol, 4)] = data[:, range(0, maxCol-4, 4)]

    # 计算TTM值
    ttmData = data + lastYearData - quarterYearBeforeData

    # 缺失值填充为前一刻值
    ttmData = pd.DataFrame(ttmData).fillna(method='ffill', axis=1)
    ttmData = np.array(ttmData)

    # 返回结果
    return ttmData

# 采用回归方法计算复合增长率
# 输入: data-财报数据, window-回归需要的窗口长度(单位为年), negaOpt-分母端负值处理(abs表示取绝对值, nan表示置为空值)
# 输出: growth-复合增长率
def YearRegressionGrowth(data, window, negaOpt):
    """
    采用回归方法计算复合增长率
    ### Parameters
    data(ndarray)-财报数据, window(int)-回归需要的窗口长度(单位为年), negaOpt(String)-分母端负值处理(abs表示取绝对值, nan表示置为空值)
    ### Returns
    growth(ndarray)-复合增长率
    """
    # 抽取年频数据
    maxCol = data.shape[1]
    yearlyData = data[:, range(0, maxCol, 4)]

    # 初始化
    yearlyBeta = np.full(np.shape(yearlyData), np.nan)
    yearlyMean = np.full(np.shape(yearlyData), np.nan)

    # 遍历每个年度, 计算复合增长率
    for t in range(window-1, yearlyData.shape[1]):
        # 获取窗口内数据
        sample = yearlyData[:, (t-window+1):t+1]
        # 计算回归系数
        yearlyBeta[:, [t]] = (np.dot(sample, np.arange(1, window+1)-(1+window)/2) / ((window+1)*(window-1)/12) / window)[:, np.newaxis]
        # 计算窗口内均值，并对负值进行处理
        tmpMean = np.mean(sample, axis=1)
        if negaOpt == 'nan':
            # 当分母为负时置nan
            tmpMean[tmpMean<0] = np.nan
        else:
            # 当分母为负时取绝对值
            tmpMean = np.abs(tmpMean)
        yearlyMean[:, [t]] = tmpMean[:, np.newaxis]
    
    # 计算增长率
    yearlyGrowth = yearlyBeta / yearlyMean
    yearlyGrowth[np.isinf(yearlyGrowth)] = np.nan

    # 将年频数据扩充至季频维度
    growth = np.full(data.shape, np.nan)
    growth[:, range(0, growth.shape[1], 4)] = yearlyGrowth

    # 返回结果
    return growth

## 风格因子计算
# 规模因子(Size)计算
# 输入: paramSet-参数集合, closePrice-股票日频原始收盘价矩阵, shareInfo-股票市值矩阵
# 输出: expo-规模因子暴露矩阵
def SizeFactorExpo(paramSet, closePrice, shareInfo):
    """
    规模因子(Size)计算
    ### Parameters
    paramSet(ParamSet)-参数集合, closePrice(ndarray)-股票日频原始收盘价矩阵, shareInfo(ndarray)-股票市值矩阵
    ### Returns
    expo(ndarray)-规模因子暴露矩阵
    """
    # 计时器启动
    print('规模因子暴露计算...')
    timeStart = time.clock()

    # 转为np数组
    closePrice = np.array(closePrice)
    shareInfo = np.array(shareInfo)

    # 计算股票市值
    vTMC = closePrice * shareInfo

    # 计算LNCAP因子
    vLNCAP = np.log(vTMC)

    # 加权合成因子暴露
    expo = paramSet.Size.LNCAPweight * vLNCAP

    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart))

    # 返回规模因子暴露
    return expo

# Beta因子计算
# 输入: paramSet-参数集合, closeAdj-股票日频复权收盘价, indexClose-指数收盘值
# 输出: expo-Beta因子暴露矩阵, volResidual-残差波动率矩阵
def BetaFactorExpo(paramSet, closeAdj, indexClose):
    """
    Beta因子计算
    ### Parameters
    paramSet(ParamSet)-参数集合, closeAdj-股票日频复权收盘价, indexClose-指数收盘值
    ### Returns
    expo-Beta因子暴露矩阵, volResidual-残差波动率矩阵
    """
    # 计时器启动
    print('Beta因子暴露计算...')
    timeStart = time.clock()

    # 转为np数组
    # closeAdj的shape[0]表示股票数,shape[1]表示天数
    closeAdj = np.array(closeAdj)
    indexClose = np.array(indexClose)

    # 时窗长度
    timeWindow = paramSet.Beta.timeWindow
    # 权重指数衰减的半衰期
    halfLifeDays = paramSet.Beta.halfLife

    # 生成权重向量
    dampling = np.power(1/2, np.arange(timeWindow, 0, -1) / halfLifeDays)

    # 计算个股收益率序列
    stockReturn = np.diff(closeAdj) / closeAdj[:, :(closeAdj.shape[1]-1)]
    stockReturn = np.concatenate((np.full((closeAdj.shape[0], 1), np.nan), stockReturn), axis=1)

    # 计算目标指数收益率序列
    indexReturn = np.diff(indexClose) / indexClose[:, :(indexClose.shape[1]-1)]
    indexReturn = np.concatenate(([[np.nan]], indexReturn), axis=1)

    # 初始化
    beta = np.full(np.shape(closeAdj), np.nan)
    alpha = np.full(np.shape(closeAdj), np.nan)
    volResidual = np.full(np.shape(closeAdj), np.nan)

    # 计算因子值
    for i in range(np.max([timeWindow+1, paramSet.updateTBegin])-1, closeAdj.shape[1]):
        # 获取窗口期内个股收益率,指数收益率(深拷贝防止对原矩阵)
        rStockWin = deepcopy(stockReturn[:, i-timeWindow+1:i+1]).reshape((stockReturn.shape[0], timeWindow))
        rIndexWin = deepcopy(indexReturn[:, i-timeWindow+1:i+1]).reshape((indexReturn.shape[0], timeWindow))

        ## 获取并调整权重向量
        # 将rIndexWin中对应的为nan的部分置0
        weight = deepcopy(dampling).reshape((1, dampling.shape[0]))
        weight[np.isnan(rIndexWin)] = 0
        # 张成权重向量
        weight = np.tile(weight, (rStockWin.shape[0], 1))
        # 将rStockWin中对应的为nan的部分置0
        weight[np.isnan(rStockWin)] = 0

        # 将因变量中的空值设为0
        rIndexWin[np.isnan(rIndexWin)] = 0

        # 将自变量中的空值设为0
        rStockWin[np.isnan(rStockWin)] = 0

        # 计算WLS回归解析
        sumWeight = np.sum(weight, axis=1)
        sumWeight = sumWeight.reshape((sumWeight.shape[0], 1))
        weightVarY = rStockWin * weight
        sumWeightVarY = np.sum(weightVarY, axis=1)
        sumWeightVarY = sumWeightVarY.reshape((sumWeightVarY.shape[0], 1))
        sumWeightVarX = np.dot(weight, rIndexWin.T)
        meanVarX = sumWeightVarX / sumWeight
        temp1 = np.dot(weightVarY, rIndexWin.T) - sumWeightVarY * meanVarX
        temp2 = np.dot(weight, np.power(rIndexWin, 2).T) - np.power(sumWeightVarX, 2) / sumWeight
        beta[:, [i]] = temp1 / temp2
        alpha[:, [i]] = sumWeightVarY / sumWeight - beta[:, [i]] * meanVarX

        # 计算残差波动率,后续波动率因子计算中需要用到
        betaResidual = rStockWin - np.dot(beta[:, [i]], rIndexWin) - np.tile(alpha[:, [i]], (1, timeWindow))
        volResidual[:, [i]] = WeightedColStd(betaResidual, dampling)
    
    # 加权合成因子暴露
    expo = paramSet.Beta.BETAweight * beta

    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart)) 

    # 返回Beta因子暴露和残差波动率
    return [expo, volResidual]

# 动量(Momentum)因子计算
# 输入: paramSet-参数集合, closeAdj-股票日频原始收盘价矩阵
# 输出: expo-动量因子暴露矩阵
def MonmentumFactorExpo(paramSet, closeAdj):
    """
    动量(Momentum)因子计算
    ### Parameters
    paramSet(ParamSet)-参数集合, closeAdj(ndarray)-股票日频原始收盘价矩阵
    ### Returns
    expo(ndarray)-动量因子暴露矩阵
    """
    # 计时器启动
    print('动量因子暴露计算...')
    timeStart = time.clock()

    # 转为np数组
    closeAdj = np.array(closeAdj)

    # 时窗长度
    timeWindow = paramSet.Mmnt.timeWindow
    # 动量计算需要剔除最近一段区间
    lag = paramSet.Mmnt.lag
    # 半衰期长度
    halfLifeDays = paramSet.Mmnt.halfLife

    # 计算个股收益率序列
    stockReturn = np.diff(closeAdj) / closeAdj[:, :(closeAdj.shape[1]-1)]
    stockReturn = np.concatenate((np.full((closeAdj.shape[0], 1), np.nan), stockReturn), axis=1)

    # 计算个股对数收益率序列并将nan值补0
    stockLogReturn = np.log(stockReturn+1)
    stockLogReturnIsNaN = np.isnan(stockLogReturn)
    stockLogReturn[stockLogReturnIsNaN] = 0

    # 计算半衰期加权矩阵
    weightMatrix = CalcMovingWeightMatrix(stockLogReturn, timeWindow, halfLifeDays)
    
    # 计算加权动量因子计算结果
    vRSTR = np.full(np.shape(stockLogReturn), np.nan)
    vRSTR[:, timeWindow+lag-1:] = np.dot(stockLogReturn, weightMatrix[:, :weightMatrix.shape[1]-lag])

    # 需要把无效位置置为nan
    tempJudgeNaN = np.dot(stockLogReturnIsNaN, weightMatrix[:, :weightMatrix.shape[1]-lag])
    tempJudgeNaN[tempJudgeNaN>0] = 1
    RSTRIsNaN = np.concatenate((np.ones((stockReturn.shape[0], timeWindow+lag-1), dtype=np.bool), tempJudgeNaN.astype(np.bool)), axis=1)
    vRSTR[RSTRIsNaN] = np.nan
    
    # 加权合成因子暴露
    expo = paramSet.Mmnt.RSTRweight * vRSTR

    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart)) 

    # 返回动量因子暴露
    return expo

# 波动率(Residual Volatility)因子计算
# 输入: paramSet-参数集合, closeAdj-股票日频原始收盘价矩阵, volResidual-Beta因子计算中得到的个股残差波动率
# 输出: expo-波动率因子暴露
def ResidVolaFactorExpo(paramSet, closeAdj, volResidual):
    """
    波动率(Residual Volatility)因子计算
    ### Parameters
    paramSet(ParamSet)-参数集合, closeAdj(ndarray)-股票日频原始收盘价矩阵, volResidual(ndarray)-Beta因子计算中得到的个股残差波动率
    ### Returns
    expo(ndarray)-波动率因子暴露
    """
    # 计时器启动
    print('波动率因子暴露计算...')
    timeStart = time.clock()

    # 转为np数组
    closeAdj = np.array(closeAdj)

    # 计算个股收益率序列
    stockReturn = np.diff(closeAdj) / closeAdj[:, :(closeAdj.shape[1]-1)]
    stockReturn = np.concatenate((np.full((closeAdj.shape[0], 1), np.nan), stockReturn), axis=1)

    ## 计算DASTD(历史波动率)
    # 参数设置
    timeWinDASTD = paramSet.ResidVola.DASTDtimeWindow
    halfLifeDASTD = paramSet.ResidVola.DASTDhalfLife
    nanOpt = paramSet.ResidVola.nanOpt

    # 忽略nan标志符,默认为false
    nanFlag = True if nanOpt == 'omitnan' else False

    # 计算权重向量
    damplingDASTD = np.power(1/2, np.arange(timeWinDASTD, 0, -1) / halfLifeDASTD)
    damplingDASTD = damplingDASTD / np.sum(damplingDASTD)

    # 初始化DASTD
    vDASTD = np.full(np.shape(closeAdj), np.nan)

    # 遍历每个截面
    for i in range(np.max([timeWinDASTD+1, paramSet.updateTBegin])-1, closeAdj.shape[1]):
        # 个股区间收益率
        returnList = deepcopy(stockReturn[:, i-timeWinDASTD+1:i+1]).reshape((stockReturn.shape[0], timeWinDASTD))
        # 加权波动率
        vDASTD[:, [i]] = WeightedColStd(returnList, damplingDASTD, nanFlag)
    
    ## 计算CMRA(历史收益率的波动幅度)
    # 参数设置
    monthNumCMRA = paramSet.ResidVola.CMRAtimeWindow
    dayNumOfMonth = paramSet.ResidVola.CMRAdayNumOfMonth
    negaOpt = paramSet.ResidVola.negaOpt

    # 计算窗长
    timeWinCMRA = monthNumCMRA * dayNumOfMonth

    # 初始化CMRA
    vCMRA = np.full(np.shape(closeAdj), np.nan)

    # 计算CMRA
    for i in range(np.max([timeWinCMRA+1, paramSet.updateTBegin])-1, closeAdj.shape[1]):
        # 获取窗口期内的月频下表索引
        mIndex = np.arange(i-timeWinCMRA, i+1, dayNumOfMonth)
        # 1-T月的对数收益累积
        returnCumMonth = np.log(closeAdj[:, mIndex[1:]] / (closeAdj[:, mIndex[0]])[:, np.newaxis])
        # 累积月收益率z(t)<=-1的负值调整
        returnCumMonth[returnCumMonth<=-1] = negaOpt
        # 计算累积收益率的波动幅度
        vCMRA[:, [i]] = (np.log(1+np.max(returnCumMonth, axis=1)) - np.log(1+np.min(returnCumMonth, axis=1)))[:, np.newaxis]

    ## 获取HSIGMA(Beta因子计算中线性回归残差项的标准差)
    vHSIGMA = volResidual

    # 获取因子权重
    weightDASTD = paramSet.ResidVola.DASTDweight
    weightCMRA = paramSet.ResidVola.CMRAweight
    weightHSIGMA = paramSet.ResidVola.HSIGMAweight

    # 加权合成因子暴露
    expo = weightDASTD*vDASTD + weightCMRA*vCMRA + weightHSIGMA*vHSIGMA

    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart))

    # 返回波动率因子暴露
    return expo

# 非线性规模(Non-linear Size)因子计算
# 输入: paramSet-参数集合, sizeFactorExpo-规模因子暴露矩阵, circulationMarketVal-流通市值, indusFactorExpo-行业因子暴露矩阵, validStockMatrix-个股有效状态标记
# 输出: expo-非线性规模因子暴露
def NonLinearFactorExpo(paramSet, sizeFactorExpo, circulationMarketVal, indusFactorExpo, validStockMatrix):
    """
    非线性规模(Non-linear Size)因子计算
    ### Parameters
    paramSet(ParamSet)-参数集合, sizeFactorExpo(ndarray)-规模因子暴露矩阵, circulationMarketVal(ndarray)-流通市值,
    indusFactorExpo(ndarray)-行业因子暴露矩阵, validStockMatrix(ndarray)-个股有效状态标记
    ### Returns
    expo(ndarray)-非线性规模因子暴露
    """
    # 计时器启动
    print('非线性因子暴露计算...')
    timeStart = time.clock()

    # 转为np数组
    sizeFactorExpo = np.array(sizeFactorExpo)
    circulationMarketVal = np.array(circulationMarketVal)
    indusFactorExpo = np.array(indusFactorExpo)
    validStockMatrix = np.array(validStockMatrix)

    # 对流通市值进行修正
    # 按分位数计算阈值
    threshold = np.apply_along_axis(lambda x: prctile_matlab(x, paramSet.Global.mktBlockRatio), axis=0, arr=circulationMarketVal)
    threshold = threshold[np.newaxis, :]
    # 修正过大市值
    circulationMarketVal = np.minimum(circulationMarketVal, threshold)

    # 计算标准化的市值因子暴露
    standSizeFactorExpo = Standardize(paramSet, sizeFactorExpo, circulationMarketVal, indusFactorExpo, validStockMatrix)

    # 将规模因子的三次方对规模因子正交, 取残差作为非线性规模因子
    targetFactor = np.power(standSizeFactorExpo, 3)
    vNLSIZE = Orthogonalize(targetFactor, standSizeFactorExpo, circulationMarketVal, validStockMatrix)

    # 加权合成因子暴露
    expo = paramSet.NonLinear.NLSIZEweight * vNLSIZE

    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart))

    # 返回非线性规模因子暴露
    return expo

# 账面市值比(Book to Price)因子计算
# 输入: paramSet-参数集合, daily_info-本地日频数据, quarterly_info-本地季频数据, useQuarterLoc-每天能看到的最新财报索引
# 输出: expo-账面市值比因子暴露
def BooktoPriceFactorExpo(paramSet, daily_info, quarterly_info, useQuarterLoc):
    """
    账面市值比(Book to Price)因子计算
    paramSet(ParamSet)-参数集合, daily_info(Dict)-本地日频数据, quarterly_info(Dict)-本地季频数据, useQuarterLoc(ndarray)-每天能看到的最新财报索引
    expo(ndaarray)-账面市值比因子暴露
    """
    # 计时器启动
    print('账面市值比因子暴露计算...')
    timeStart = time.clock()

    # 计算市值
    vMC = None
    if paramSet.BktoPrc.mrkCapType == 'total':
        vMC = daily_info['close'][0][0] * daily_info['total_shares'][0][0]
    else:
        vMC = daily_info['close'][0][0] * daily_info['float_a_shares'][0][0]
    
    # 获取季报索引
    quarterLoc = None
    if paramSet.BktoPrc.rptFreq == 'annual':
        quarterLoc = useQuarterLoc[0]
    else:
        quarterLoc = useQuarterLoc[1]
    
    # 获取每天账面价值
    vCE = np.full(np.shape(vMC), np.nan)
    tot_equity = quarterly_info['tot_equity'][0][0]
    for iStock in range(0, vCE.shape[0]):
        # 每天能看到的财报索引
        qLoc = quarterLoc[iStock, :]
        # 填充日频信息
        vCE[iStock, ~np.isnan(qLoc)] = tot_equity[iStock, (qLoc[~np.isnan(qLoc)]).astype('int')]
    
    # 计算账面市值比BTOP
    vBTOP = vCE / vMC

    # 加权合成因子暴露
    expo = paramSet.BktoPrc.BTOPweight * vBTOP

    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart))

    # 返回账面市值比因子暴露
    return expo

# 流动性(Liquidity)因子计算
# 输入: paramSet-参数集合, daily_info_turn-换手率数据
# 输出: expo-流动性因子暴露
def LiquidityFactorExpo(paramSet, daily_info_turn):
    # 计时器启动
    print('流动性因子暴露计算...')
    timeStart = time.clock()

    # 转为np数组
    daily_info_turn = np.array(daily_info_turn)

    # 日期维度
    dayNum = daily_info_turn.shape[1]

    ## 计算STOM(过去一个月的流动性)
    # 一个月包含的天数
    timeWin = paramSet.Liquidity.dayNumOfMonth

    # 是否忽略缺失值
    nanOpt = paramSet.Liquidity.nanOpt

    # 初始化
    vSTOM = np.full(daily_info_turn.shape, np.nan)

    # 计算STOM
    for i in range(np.max([timeWin, paramSet.updateTBegin])-1, dayNum):
        # 计算区间内换手率累计值
        kernel = np.nanmean(daily_info_turn[:, (i+1-timeWin): i+1], axis=1) if nanOpt == 'omitnan' else np.mean(daily_info_turn[:, (i+1-timeWin): i+1], axis=1)
        kernel = kernel * timeWin
        # 取对数
        vSTOM[:, [i]] = (np.log(kernel))[:, np.newaxis]

    # 修正换手率为零的场景
    vSTOM[np.isneginf(vSTOM)] = np.nan

    ## 计算STOQ(过去一个季度的流动性)
    # 一个季度包含的月份数
    monthNumSTOQ = paramSet.Liquidity.STOQTime

    # 对应的日频窗长
    timeWinSTOQ = paramSet.Liquidity.dayNumOfMonth * monthNumSTOQ

    # 初始化
    vSTOQ = np.full(daily_info_turn.shape, np.nan)

    # 计算STOQ
    for i in range(np.max([timeWinSTOQ, paramSet.updateTBegin])-1, dayNum):
        # 计算区间内换手率累计值
        kernel = np.nanmean(daily_info_turn[:, (i+1-timeWinSTOQ): i+1], axis=1) if nanOpt == 'omitnan' else np.mean(daily_info_turn[:, (i+1-timeWinSTOQ): i+1], axis=1)
        kernel = kernel * timeWinSTOQ
        # 取对数
        vSTOQ[:, [i]] = (np.log(kernel/monthNumSTOQ))[:, np.newaxis]
    
    # 修正换手率为零的场景
    vSTOQ[np.isneginf(vSTOQ)] = np.nan

    ## 计算STOA(过去一年的流动性)
    monthNumSTOA = paramSet.Liquidity.STOATime

    # 对应的日频窗长
    timeWinSTOA = paramSet.Liquidity.dayNumOfMonth * monthNumSTOA

    # 初始化
    vSTOA = np.full(daily_info_turn.shape, np.nan)

    # 计算STOA
    for i in range(np.max([timeWinSTOA, paramSet.updateTBegin])-1, dayNum):
        # 计算区间内换手率累计值
        kernel = np.nanmean(daily_info_turn[:, (i+1-timeWinSTOA): i+1], axis=1) if nanOpt == 'omitnan' else np.mean(daily_info_turn[:, (i+1-timeWinSTOA): i+1], axis=1)
        kernel = kernel * timeWinSTOA
        # 取对数
        vSTOA[:, [i]] = (np.log(kernel/monthNumSTOA))[:, np.newaxis]
    
    # 修正换手率为零的场景
    vSTOA[np.isneginf(vSTOA)] = np.nan

    # 加权合成因子暴露
    expo = paramSet.Liquidity.STOMweight * vSTOM + paramSet.Liquidity.STOQweight * vSTOQ + paramSet.Liquidity.STOAweight * vSTOA

    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart))

    # 返回流动性因子暴露
    return expo

# 盈利(Earning Yield)因子计算
# 输入: paramSet-参数集合, daily_info-本地日频数据, quarterly_info-本地季频信息, useQuarterLoc-每天能看到的最新财报索引
# 输出: expo-盈利因子暴露
def EarningYieldFactorExpo(paramSet, daily_info, quarterly_info, useQuarterLoc):
    # 计时器启动
    print('盈利因子暴露计算...')
    timeStart = time.clock()

    # 计算市值
    vMC = None
    if paramSet.EarningYield.mrkCapType == 'total':
        vMC = np.array(daily_info['close'][0][0]) * np.array(daily_info['total_shares'][0][0])
    else:
        vMC = np.array(daily_info['close'][0][0]) * np.array(daily_info['float_a_shares'][0][0])
    
    # 获取季报索引
    quarterLoc = None
    if paramSet.EarningYield.rptFreq == 'annual':
        quarterLoc = useQuarterLoc[0]
    else:
        quarterLoc = useQuarterLoc[1]

    ## 计算ETOP(过去12个月的市盈率)
    # 获取归母净利润TTM值
    vNNP = TransformTTM(quarterly_info['np_belongto_parcomsh'][0][0])

    # 按照财报发布日期, 将季频数据扩充至日频
    vE = np.full(np.shape(vMC), np.nan)
    for iStock in range(0, vMC.shape[0]):
        # 每天能看到的财报索引
        qLoc = quarterLoc[iStock, :]
        # 填充日频信息
        vE[iStock, ~np.isnan(qLoc)] = vNNP[iStock, (qLoc[~np.isnan(qLoc)]).astype('int')]

    # 计算ETOP
    vETOP = vE / vMC

    ## 计算CETOP(过去12个月的经营性净现金流与市值的比值)
    # 获取经营性净现金流TTM值
    vNCF = TransformTTM(quarterly_info['net_cash_flows_oper_act'][0][0])

    # 按照财报发布日期, 将季频数据扩充至日频
    vCE = np.full(np.shape(vMC), np.nan)
    for iStock in range(0, vMC.shape[0]):
        # 每天能看到的财报索引
        qLoc = quarterLoc[iStock, :]
        # 填充日频信息
        vCE[iStock, ~np.isnan(qLoc)] = vNCF[iStock, (qLoc[~np.isnan(qLoc)]).astype('int')]
    
    # 计算CETOP
    vCETOP = vCE / vMC

    # 加权合成因子暴露
    expo = (paramSet.EarningYield.CETOPweight * vCETOP + paramSet.EarningYield.ETOPweight * vETOP) / (paramSet.EarningYield.CETOPweight + paramSet.EarningYield.ETOPweight)

    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart))

    # 返回盈利因子暴露
    return expo

# 成长(Growth)因子暴露
# 输入: paramSet-参数集合, daily_info-本地日频数据, quarterly_info-本地季频信息, useQuarterLoc-每天能看到的最新财报索引
# 输出: expo-成长因子暴露
def GrowthFactorExpo(paramSet, daily_info, quarterly_info, useQuarterLoc):
    # 计时器启动
    print('成长因子暴露计算...')
    timeStart = time.clock()

    # 获取季报索引
    quarterLoc = useQuarterLoc[0]

    # 复合增长率计算的时间窗长(单位为年)
    window = paramSet.Growth.yearWindow

    # 复合增长率计算时分母端负值处理
    negaOpt = paramSet.Growth.negaOpt

    ## 计算EGRO(过去5年企业归属母公司净利润的复合增长率)
    # 获取复合增长率数据
    profitGrowth = YearRegressionGrowth(quarterly_info['net_profit_is'][0][0], window, negaOpt)

    # 填充至日频
    vEGRO = np.full(np.shape(daily_info['close'][0][0]), np.nan)
    for iStock in range(0, vEGRO.shape[0]):
        # 每天能看到的财报索引
        qLoc = quarterLoc[iStock, :]
        # 填充日频信息
        vEGRO[iStock, ~np.isnan(qLoc)] = profitGrowth[iStock, (qLoc[~np.isnan(qLoc)]).astype('int')]
    
    ## 计算SGRO(过去5年企业营业总收入的复合增长率)
    operGrowth = YearRegressionGrowth(quarterly_info['tot_oper_rev'][0][0], window, negaOpt)

    # 填充至日频
    vSGRO = np.full(np.shape(daily_info['close'][0][0]), np.nan)
    for iStock in range(0, vSGRO.shape[0]):
        # 每天能看到的财报索引
        qLoc = quarterLoc[iStock, :]
        # 填充日频信息
        vSGRO[iStock, ~np.isnan(qLoc)] = operGrowth[iStock, (qLoc[~np.isnan(qLoc)]).astype('int')]
    
    # 加权合成因子暴露
    expo = (paramSet.Growth.EGROweight * vEGRO + paramSet.Growth.SGROweight * vSGRO) / (paramSet.Growth.EGROweight + paramSet.Growth.SGROweight)

    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart))

    # 返回成长因子暴露
    return expo

# 杠杆(Leverage)因子暴露
# 输入: paramSet-参数集合, daily_info-本地日频数据, quarterly_info-本地季频信息, useQuarterLoc-每天能看到的最新财报索引
# 输出: expo-成长因子暴露
def LeverageFactorExpo(paramSet, daily_info, quarterly_info, useQuarterLoc):
    # 计时器启动
    print('杠杆因子暴露计算...')
    timeStart = time.clock()

    # 计算总市值
    vMC = daily_info['close'][0][0] * daily_info['total_shares'][0][0]

    # 获取季报索引
    quarterLoc = None
    if paramSet.Leverage.rptFreq == 'annual':
        quarterLoc = useQuarterLoc[0]
    else:
        quarterLoc = useQuarterLoc[1]
    
    # 获取日频财务数据
    # 长期负债
    vLD = np.full(vMC.shape, np.nan)
    # 总负债
    vTD = np.full(vMC.shape, np.nan)
    # 总资产
    vTA = np.full(vMC.shape, np.nan)
    # 总权益
    vBE = np.full(vMC.shape, np.nan)
    for iStock in range(0, vMC.shape[0]):
        # 每天能看到的财报索引
        qLoc = quarterLoc[iStock, :]
        # 数据填充
        vLD[iStock, ~np.isnan(qLoc)] = (quarterly_info['tot_non_cur_liab'][0][0])[iStock, (qLoc[~np.isnan(qLoc)]).astype('int')]
        vTD[iStock, ~np.isnan(qLoc)] = (quarterly_info['tot_liab'][0][0])[iStock, (qLoc[~np.isnan(qLoc)]).astype('int')]
        vTA[iStock, ~np.isnan(qLoc)] = (quarterly_info['tot_assets'][0][0])[iStock, (qLoc[~np.isnan(qLoc)]).astype('int')]
        vBE[iStock, ~np.isnan(qLoc)] = (quarterly_info['tot_equity'][0][0])[iStock, (qLoc[~np.isnan(qLoc)]).astype('int')]
    
    # 计算MLEV
    vMLEV = 1 + vLD / vMC

    # 计算DTOA
    vDTOA = vTD / vTA

    # 计算BLEV
    vBLEV = 1 + vLD / vBE

    # 加权合成因子暴露
    expo = paramSet.Leverage.MLEVweight * vMLEV + paramSet.Leverage.DTOAweight * vDTOA + paramSet.Leverage.BLEVweight * vBLEV

    # 计时器结束
    timeEnd = time.clock()
    print('计算耗时:{}Seconds'.format(timeEnd - timeStart))

    # 返回杠杆因子暴露
    return expo
    
