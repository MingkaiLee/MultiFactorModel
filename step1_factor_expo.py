# 多因子风险模型第一步: 生成因子暴露、因子收益率、特质收益率
# 1) 默认采用增量运行模式, 回溯窗长为ParamSet.backLength参数, 一般应用场景为月度更新，所以默认回溯长度为23个交易日
# 2) 如需重头开始运行, 可以将ParamSet.backLength设置为-1
# 导入包
import numpy as np
from scipy.io import loadmat
import gc
# 导入其他部分代码
# 欲导入部分的路径置入
import sys
sys.path.append("module_1/utils")
sys.path.append("module_1/factor")
from ParamSet import *
from CalcValidStockMatrix import *
from CalcUseQuarterLoc import *
from FactorExpoCalculator import *
from FactorProcess import *
from FactorReturn import *
from SpecialReturn import *
from MergeAndSaveFactorData import *


def step1_factor_expo(stock_data_dir, index_data_dir):
    # 原始信息读取
    # 获取股票基本信息
    basic_info = loadmat(stock_data_dir, variable_names=['basic_info'])['basic_info']
    # 获取股票的日频信息
    daily_info = loadmat(stock_data_dir, variable_names=['daily_info'])['daily_info']
    # 获取股票的月频信息
    monthly_info = loadmat(stock_data_dir, variable_names=['monthly_info'])['monthly_info']
    # 读取股票的季频信息
    quarterly_info = loadmat(stock_data_dir, variable_names=['quarterly_info'])['quarterly_info']

    # 生成全局参数
    paramSet = ParamSet()

    # 计算起始索引
    paramSet.CalcUpdateTBegin(daily_info)

    # 获取有效股票标记矩阵
    validStockMatrix = CalcValidStockMatrix(basic_info, daily_info)

    # 分别获取每个截面上能看到的最新年报和最新季报位置索引
    useQuarterLocAnnual = CalcUseQuarterLoc(daily_info['dates'][0][0], quarterly_info['issue_date'][0][0], 'annual')
    useQuarterLocQuarterly = CalcUseQuarterLoc(daily_info['dates'][0][0], quarterly_info['issue_date'][0][0], 'quarterly')
    useQuarterLoc = [useQuarterLocAnnual, useQuarterLocQuarterly]

    # 计算风格因子暴露
    # 规模因子
    # 读取股票原始收盘价信息
    closePrice = daily_info['close'][0][0]
    # 读取股票的股本信息
    shareInfo = daily_info['total_shares'][0][0] if paramSet.Size.mrkCapType == 'total' else daily_info['float_a_shares'][0][0]
    sizeFactorExpo = SizeFactorExpo(paramSet, closePrice, shareInfo)

    # Beta因子
    # 读取股票的复权收盘价信息
    closeAdj = daily_info['close_adj'][0][0]
    # 读取指数信息
    indexInfo = loadmat(index_data_dir, variable_names=['index_data'])['index_data']
    indexClose = (((indexInfo[paramSet.Beta.indexName])[0][0])['close'])[0][0]
    betaFactorExpo, volResidual = BetaFactorExpo(paramSet, closeAdj, indexClose)

    # 动量因子
    mmntFactorExpo = MonmentumFactorExpo(paramSet, closeAdj)
    
    # 波动率因子
    residVolaFactorExpo = ResidVolaFactorExpo(paramSet, closeAdj, volResidual)
    
    # 非线性市值因子
    # 计算流通市值
    shareInfo = daily_info['float_a_shares'][0][0]
    circulationMarketVal = closePrice * shareInfo
    # 获取行业因子暴露
    indusExpo = daily_info['cs_indus_code'][0][0]
    nonlinearFactorExpo = NonLinearFactorExpo(paramSet, sizeFactorExpo, circulationMarketVal, indusExpo, validStockMatrix)

    # 账面市值比因子
    btpFactorExpo = BooktoPriceFactorExpo(paramSet, daily_info, quarterly_info, useQuarterLoc)

    # 流动性因子
    liquidityFactorExpo = LiquidityFactorExpo(paramSet, daily_info['turn'][0][0])

    # 盈利因子
    earningYieldFactorExpo = EarningYieldFactorExpo(paramSet, daily_info, quarterly_info, useQuarterLoc)

    # 成长因子
    growthFactorExpo = GrowthFactorExpo(paramSet, daily_info, quarterly_info, useQuarterLoc)
    
    # 杠杆因子
    leverageFactorExpo = LeverageFactorExpo(paramSet, daily_info, quarterly_info, useQuarterLoc)

    # 将10大类风格因子的因子暴露组成字典
    styleExpo = {'size': sizeFactorExpo, 'beta': betaFactorExpo, 'momentum': mmntFactorExpo, 'residVola': residVolaFactorExpo,
    'nonLinear': nonlinearFactorExpo, 'booktoPrice': btpFactorExpo, 'liquidity': liquidityFactorExpo, 'earningYield': earningYieldFactorExpo,
    'growth': growthFactorExpo, 'leverage': leverageFactorExpo}

    # 因子暴露正交化(返回增量更新部分)
    FactorProcess(paramSet, daily_info, styleExpo, indusExpo, validStockMatrix)

    # 因子收益率、残差收益率计算
    # 计算因子收益率和因子暴露(返回增量更新部分)
    factorReturn, factorExpo = FactorReturn(paramSet, daily_info, styleExpo, indusExpo, validStockMatrix)

    # 计算残差收益率(返回增量更新部分)
    specialReturn = SpecialReturn(paramSet, daily_info, factorExpo, factorReturn)

    # 清除无关变量
    del styleExpo, indusExpo, daily_info, monthly_info, quarterly_info
    gc.collect()

    # 将增量计算结果与本地存储结果合并
    factorExpo, factorReturn, specialReturn = MergeAndSaveFactorData(paramSet, factorExpo, factorReturn, specialReturn)

    print('Step1 finished.')

    
# 指数信息路径
param_index_data_dir = 'module_1/data/index_data.mat'
# 股票信息路径
param_stock_data_dir = 'module_1/data/stock_data.mat'


if __name__ == '__main__':
    step1_factor_expo(param_stock_data_dir, param_index_data_dir)