# 导入包
from scipy.io import loadmat
import numpy as np


# 全局参数
class Global:
    def __init__(self, backLength=-1, capSqrtFlag=1, mktBlockRatio=95, thDataTooFew=0, thZscore=3, save_path='module_1/result/') -> None:
        # 每次更新数据后,回溯运行多少天,设置为-1就表示初始化模式
        self.backLength = backLength
        # 市值加权权重是否取平方根: 1-取; 0-不取
        self.capSqrtFlag = capSqrtFlag
        # 修正大市值股票的修正分位数
        self.mktBlockRatio = mktBlockRatio
        # 缺失值填充行业均值时,有效数据量下限
        self.thDataTooFew = thDataTooFew
        # Z-score去极值中的阈值(中位数的倍数)
        self.thZscore = thZscore
        # 中间结果保存路径
        self.save_path = save_path

        # 固定参数,不可修改
        # 风格因子个数
        self.styleFactorNum = 10
        # 行业因子个数
        self.indusFactorNum = 30
        # 国家因子个数
        self.countryFactorNum = 1

# 因子暴露计算相关参数
# 规模因子(Size)
class Size:
    def __init__(self, mrkCapType='total', LNCAPweight=1.0) -> None:
        # 'total'-总市值; 'float'-流通市值 
        self.mrkCapType = mrkCapType
        # 细类因子合成权重
        self.LNCAPweight = LNCAPweight

# Beta因子(Beta)
class Beta:
    def __init__(self, timeWindow=252, halfLife=63, indexName='ZZQZ', BETAweight=1.0) -> None:
        # 时窗长度
        self.timeWindow = timeWindow
        # 半衰期长度
        self.halfLife = halfLife
        # 回归基准
        self.indexName = indexName
        # 细类因子合成权重
        self.BETAweight = BETAweight

# 动量因子(Momentum)
class Mmnt:
    def __init__(self, timeWindow=504, halfLife=126, lag=21, RSTRweight=1.0) -> None:
        # 时窗长度
        self.timeWindow = timeWindow
        # 半衰期长度
        self.halfLife = halfLife
        # 剔除近1个月样本
        self.lag = lag
        # 细类因子合成权重
        self.RSTRweight = RSTRweight

# 波动率(Residual Volatility)
class ResidVola:
    def __init__(self, nanOpt='omitnan', negaOpt=-0.99, indexName='ZZQZ', DASTDtimeWindow=252, DASTDhalfLife=42, CMRAtimeWindow=12, CMRAdayNumOfMonth=21, HSIGMAtimeWindow=252, 
        HSIGMAhalfLife=63, HSIGMAregressWindow=252, HSIGMAregressHalfLife=63, DASTDweight=0.74, CMRAweight=0.16, HSIGMAweight=0.10) -> None:
        # DASTD与HSIGMA计算中是否忽略nan: 'omitnan'-忽略nan; 'includenan'-不忽略nan
        self.nanOpt = nanOpt
        # CMRA计算中,累积T月对数收益Z(T)<=-1时的负值调整
        self.negaOpt = negaOpt
        # 回归基准
        self.indexName = indexName
        # DASTD时窗长度
        self.DASTDtimeWindow = DASTDtimeWindow
        # DASTD权重指数衰减的半衰期长度
        self.DASTDhalfLife = DASTDhalfLife
        # CRMA时窗长度
        self.CMRAtimeWindow = CMRAtimeWindow
        # CMRA求和时间区段长度
        self.CMRAdayNumOfMonth = CMRAdayNumOfMonth
        # HSIGMA时窗长度
        self.HSIGMAtimeWindow = HSIGMAtimeWindow
        # HSIGMA权重指数衰减的半衰期长度
        self.HSIGMAhalfLife = HSIGMAhalfLife
        # HSIGMA回归参数-时窗
        self.HSIGMAregressWindow = HSIGMAregressWindow
        # HSIGMA回归参数-半衰期
        self.HSIGMAregressHalfLile = HSIGMAregressHalfLife
        # DASTD权重
        self.DASTDweight = DASTDweight
        # CMRA权重
        self.CMRAweight = CMRAweight
        # HSIGMA权重
        self.HSIGMAweight = HSIGMAweight

# 非线性规模因子
class NonLinear:
    def __init__(self, NLSIZEweight=1.0) -> None:
        # NLSIZE权重
        self.NLSIZEweight = NLSIZEweight

# Book to Price
class BktoPrc:
    def __init__(self, rptFreq='annual', mrkCapType='total', BTOPweight=1.0) -> None:
        # 使用财报类型: 'annual'-只使用最新年报数据; 'quarterly'-使用最新的季频数据
        self.rptFreq = rptFreq
        # 市值类型: 'total'-总市值; 'float'-流通市值
        self.mrkCapType = mrkCapType
        # BTOP权重
        self.BTOPweight = BTOPweight

# 流动性(Liquidity)
class Liquidity:
    def __init__(self, nanOpt='omitnan', dayNumOfMonth=21, STOQTime=3, STOATime=12, STOMweight=0.35, STOQweight=0.35, STOAweight=0.30) -> None:
        # 换手率计算中是否忽略anan: 'omitnan'-忽略nan; 'includenan'-不忽略nan
        self.nanOpt = nanOpt
        # 换手率计算区间(月)长度
        self.dayNumOfMonth = dayNumOfMonth
        # STOQ换手率计算的时间长度(月)
        self.STOQTime = STOQTime
        # STOA换手率计算的时间长度(月)
        self.STOATime = STOATime
        # 因子合成中STOM的权重
        self.STOMweight = STOMweight
        # 因子合成中STOQ的权重
        self.STOQweight = STOQweight
        # 因子合成中STOA的权重
        self.STOAweight = STOAweight

# Earning Yield因子
class EarningYield:
    def __init__(self, mrkCapType='total', rptFreq='quarterly', EPFWDweight=0.68, CETOPweight=0.21, ETOPweight=0.11) -> None:
        # CETOP与ETOP中的市值类型: ''total-总市值; 'float'-流通市值
        self.mrkCapType = mrkCapType
        # 使用财报类型: 'annual'-只使用最新年报数据; 'quarterly'-使用最新的季频数据
        self.rptFreq = rptFreq
        # 因子合成中EPFWD的权重(一致预期因子,实际舍去)
        self.EPFWDweight = EPFWDweight
        # 因子合成中CETOP的权重(实际为0.21/0.32=0.66)
        self.CETOPweight = CETOPweight
        # 因子合成中ETOP的权重(实际为0.11/0.32=0.34)
        self.ETOPweight = ETOPweight

# 成长性因子(Growth)
class Growth:
    def __init__(self, negaOpt='abs', yearWindow=5, EGRLFweight=0.18, EGRSFweight=0.11, EGROweight=0.24, SGROweight=0.47) -> None:
        # EGRO计算中分母为负时的调整: 'abs'-分母取绝对值; 'nan'-分母置nan
        self.negaOpt = negaOpt
        # 考察多少年的复合增长率
        self.yearWindow = yearWindow
        # 因子合成中EGRLF的权重(一致预期因子,舍去)
        self.EGRLFweight = EGRLFweight
        # 因子合成红EGRSF的权重(一致预期因子,舍去)
        self.EGRSFweight = EGRSFweight
        # 因子合成中EGRO的权重(实际为0.24/0.71=0.34)
        self.EGROweight = EGROweight
        # 因子合成中SGRO的权重(实际为0.47/0.71=0.66)
        self.SGROweight = SGROweight

# 杠杆因子(Leberrage)
class Leverage:
    def __init__(self, rptFreq='annual', MLEVweight=0.38, DTOAweight=0.35, BLEVweight=0.27) -> None:
        # 使用财报类型: 'annual'-只使用最新年报数据; 'quarterly'-使用最新季频数据
        self.rptFreq = rptFreq
        # 因子合成中MLEV的权重
        self.MLEVweight = MLEVweight
        # 因子合成中DTOA的权重
        self.DTOAweight = DTOAweight
        # 因子合成中BLEV的权重
        self.BLEVweight = BLEVweight

# 协方差矩阵估计相关参数
# 因子协方差矩阵的NW调整的参数
class NW:
    def __init__(self, tBegin=1000, timeWindow=252, timeWindowS=42, halfLife=90, dayNumOfMonth=21, d=2) -> None:
        # NW调整计算的开始日期
        self.tBegin = tBegin
        # NW调整的时窗长度
        self.timeWindow = timeWindow
        # NW调整的时窗长度(短)
        self.timeWindowS = timeWindowS
        # NW调整的指数加权半衰期
        self.halfLife = halfLife
        # NW调整的月频数据使用的倍率. 代表每月的天数
        self.dayNumOfMonth = dayNumOfMonth
        # NW调整中错位互相关的错位最大日期
        self.d = d

# 因子协方差矩阵的特征值调整参数
class EigenAdj:
    def __init__(self, mcs=3000, timeWindow=100, a=1.5) -> None:
        # 蒙特卡洛模拟次数
        self.mcs = mcs
        # 特征值调整选取的时窗长度
        self.timeWindow = timeWindow
        # 由于尖峰厚尾特征, 需对模拟风险偏差进行调整
        self.a = a

# 因子协方差矩阵的波动率调整参数
class VolRegAdj:
    def __init__(self, timeWin=252, halfLife=42) -> None:
        # 波动率偏误调整的时窗长度
        self.timeWin = timeWin
        # 波动率偏误调整的半衰期
        self.halfLife = halfLife

# 特异性协方差矩阵的NW调整的参数
class SpeNW:
    def __init__(self, tBegin=1000, timeWindow=252, halfLife=90, dayNumOfMonth=21, d=5) -> None:
        # NW调整计算的开始日期
        self.tBegin = tBegin
        # NW调整的时窗长度
        self.timeWindow = timeWindow
        # NW调整的指数加权半衰期
        self.halfLife = halfLife
        # NW调整的月频数据使用的倍率. 代表每月的天数
        self.dayNumOfMonth = dayNumOfMonth
        # NW调整中错位互相关的错位最大日期
        self.d = d
        
# 特异性协方差矩阵的结构化调整的参数
class SpeStructAdj:
    def __init__(self, timeWindow=252, e0=1.05, regressMode='WLS', factorNum=41) -> None:
        # 结构化调整的时窗长度
        self.timeWindow = timeWindow
        # 结构化调整的参数E0
        self.e0 = e0
        # 结构化调整的回归模式, 'WLS'流通市值加权, 'OLS'普通最小二乘
        self.regressMode = regressMode
        # 结构化调整的因子数目，可只选10个风格因子
        self.factorNum = factorNum

# 特异性协方差矩阵的贝叶斯压缩的参数
class SpeBiasAdj:
    def __init__(self, q=1, groupNum=10, blockRatio=95) -> None:
        # 贝叶斯压缩的系数q
        self.q = q
        # 贝叶斯压缩的市值分组数目
        self.groupNum = groupNum
        # 贝叶斯压缩的大市值调整的分位数
        self.blockRatio = blockRatio

# 特异性协方差矩阵波动率调整的参数
class SpeVolRegAdj:
    def __init__(self, timeWindow=252, halfLife=42) -> None:
        # 波动率调整的时窗长度
        self.timeWindow = timeWindow
        # 波动率调整的半衰期
        self.halfLife = halfLife

# 参数集合
class Param:
    def __init__(self) -> None:
        # 全局参数
        self.Global = Global()

        # 因子暴露计算相关参数
        # 规模因子
        self.Size = Size()
        # Beta因子
        self.Beta = Beta()
        # 动量因子
        self.Mmnt = Mmnt()
        # 波动率
        self.ResidVola = ResidVola()
        # 非线性规模因子
        self.NonLinear = NonLinear()
        # Book to Price
        self.BktoPrc = BktoPrc()
        # 流动性
        self.Liquidity = Liquidity()
        # Earning Yield因子
        self.EarningYield = EarningYield()
        # 成长性因子
        self.Growth = Growth()
        # 杠杆因子
        self.Leverage = Leverage()
        # 起始索引
        self.updateTBegin = None
        # 最新年报位置索引
        self.quarterLocAnnual = None
        # 最新季报位置索引
        self.quarterLocQuarter = None

        # 协方差矩阵估计相关参数
        # 因子协方差矩阵的NW调整
        self.nw = NW()
        # 因子协方差矩阵特征值调整
        self.EigenAdj = EigenAdj()
        # 因子协方差矩阵波动率调整
        self.volRegAdj = VolRegAdj()
        # 特异性协方差矩阵的NW调整
        self.SpeNW = SpeNW()
        # 特异性协方差矩阵的结构化调整
        self.SpeStructAdj = SpeStructAdj()
        # 特异性协方差矩阵的贝叶斯压缩
        self.SpeBiasAdj = SpeBiasAdj()
        # 特异性协方差矩阵的波动率调整
        self.SpeVolRegAdj = SpeVolRegAdj()

    
    # 获取因子暴露、因子收益、协方差估计的计算起始索引(日频信息中的索引)
    # 由于程序运行耗时较长,且历史结果不会改变,每次更新数据后计算增量部分即可
    # 本应用默认月度更新,所以每月末只需要计算这个月的新增数据即可
    def CalcUpdateTBegin(self, daily_info):
        # 当回溯长度为-1时,表示初始化模式,也即从头开始运行
        if self.Global.backLength == -1:
            self.updateTBegin = 1
        # 当回溯长度为任意正数时,表示增量模式,也即从指定位置开始运行
        # updateTBegin = len(dailyinfo.dates) - self.Global.backLength
        # 从已保存因子数据最后一天开始回溯固定天数(原代码从行情数据最后一天开始回溯固定的天数)
        else:
            factorReturn = np.load('module1_1/result/factorReturn.npy')
            self.updateTBegin = factorReturn.shape[1] - self.Global.backLength


# 获取全局参数
def ParamSet():
    res = Param()
    return res


# 测试样例
if __name__ == '__main__':
    x = ParamSet()
    print(x.Global.backLength)