# 导入包
import numpy as np


# 获取有效股票标记矩阵, 也即在每个截面上剔除无法交易的股票:
# -上市不满一年
# -退市
# -T日换手率为0
# 输入: basic_info-股票的基本信息, daily_info-股票的日频信息
# 输出: validStockMatrix-有效股票标记为1, 无效股票标记为nan
def CalcValidStockMatrix(basic_info, daily_info):
    # 获取股票和日期个数
    stockNum_dayNum = np.shape(daily_info['close'][0][0])
    # 获取日期矩阵
    dates = daily_info['dates'][0][0]
    # 获取上市日期
    ipoDate = basic_info['ipo_date'][0][0]
    # 获取退市日期
    delistDate = basic_info['delist_date'][0][0]
    # 获取股票换手率
    trun = daily_info['turn'][0][0]
    # 换手率阈值
    thresholdLimit = 1e-12

    # 初始化
    validStockMatrix = np.full(stockNum_dayNum, np.nan)

    # 遍历每支股票
    for i in range(0, stockNum_dayNum[0]):
        # 个股上市1年后开始记为有效
        beginDateNo = np.sum(dates<ipoDate[i]+365) + 1

        endDateNo = None
        # 如果股票退市计算至退市前一天, 否则截至最新日期
        if not np.isnan(delistDate[i]):
            endDateNo = np.sum(dates<delistDate[i])
        else:
            endDateNo = stockNum_dayNum[1]
        
        # 设置有效标记
        validStockMatrix[i, beginDateNo-1: endDateNo] = 1
    
    # 剔除极低换手率数据
    validStockMatrix[~(trun>=thresholdLimit)] = np.nan

    return validStockMatrix
