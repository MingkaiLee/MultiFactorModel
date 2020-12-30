# 导入包
import numpy as np
import pandas as pd


# 按照财报发布时间, 在日频日期内定位最新可使用的财报数据
# 输入: daily_dates-日频日期序列, quarterly_issue_date-个股财报真实公布时间, rptFreq-财报使用频率, 'quarterly'季报, 'annual'年报
# 输出: useQuarterLoc-每个日频日期能获取的最新财报索引
def CalcUseQuarterLoc(dailyDate, rptDate, paramRptFreq='quarterly'):
    '''

    :param dailyDate: 日频日期, 即 dailyinfo_dates, np.array (daynum,)
    :param rptDate: 财报发布日期, 即 quarterlyinfo_stm_issuingdate1, np.array（stockNum * quarterNum）
    :param paramRptFreq: 财报使用频率
                     -- ‘quarterly' - (默认选项) 使用所有财报数据，即quarterlyinfo中的所有列
                     -- ’annual‘    - 只使用年报数据，即quarterlyinfo中满足mod(t,4)==1的列

    :return: dayUseQuarterLoc：第n行t列表示第n只个股，在第t个日频日期下，可使用的最新财报在quarterlyinfo中的列数
            （stocknum * datenum)
    '''

    dailyDates = dailyDate.T
    issueDate = rptDate
    rptFreq = paramRptFreq

    # 参数
    n_stock, n_quarter = issueDate.shape
    n_dates = dailyDate.shape[1]

    # 根据使用的财报类型， 抽取相应的真实公布日期
    if rptFreq == 'annual':
        columnInterval = 4
    else:
        columnInterval = 1

    # 本地数据库中季报是从1997年年报开始存储， 也即第一列为年报
    selectColumn = np.arange(0, n_quarter, columnInterval)
    selectIssueDate = issueDate[:, selectColumn]

    # 初始化输出矩阵
    useQuarterLoc = np.full([n_stock, n_dates], np.nan)

    # 遍历每支股票
    for i_stock in range(0, n_stock):

        if np.sum(~np.isnan(selectIssueDate[i_stock, :])):

            # 获取当前股票有效的财务报表真实公布日期序列
            validColumn = np.where(~np.isnan(selectIssueDate[i_stock, :]))[0]
            validIssueDate = selectIssueDate[i_stock, validColumn]

            # 正常情况下财报公布日期是顺序的，但由于存在事后修正操作，导致财报
            # 公布日期错乱（比如2005年年报在2008年才公布），此时需要剔除无效值
            # 剔除条件：当某期季报的真实公布日期晚于后面任何一期季报公布日期
            min_val = np.array([min(validIssueDate[i:]) for i in range(len(validIssueDate))])
            valid_index = validIssueDate <= min_val
            validIssueDate = validIssueDate[valid_index]
            validColumn = validColumn[valid_index]

            # 将财报索引填入全日频日期序列中，也即找到大于公布日期的第一个点
            dailyIndex = [np.where(dailyDates >= i)[0][0] for i in validIssueDate]
            useQuarterLoc[i_stock, dailyIndex] = selectColumn[validColumn[0:len(dailyIndex)]]

    dayUseQuarterLoc = pd.DataFrame(useQuarterLoc).fillna(method='ffill', axis=1)
    useQuarterLoc = np.array(dayUseQuarterLoc)
    
    # 返回结果
    return useQuarterLoc
