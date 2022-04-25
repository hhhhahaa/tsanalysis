from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

class timeseries():
    '''
    封装了ADF检验、时序分解、趋势、周期、季节消除（因为都太短了不知道怎么分开写比较好……之前真没做过面向对象编程LOL
    '''

    def __init__(self,data,timecolname,datacolname):
        '''

        :param series: pd.Series,时序数据，index需要为时间
        :param timecolname: 时间项的列名
        :param datacolname: 时序数据的列名
        '''

        data = data.set_index(timecolname)
        data.index = pd.to_datetime(data.index)
        series = data[datacolname]
        self.data=(series)
        #self.data=np.log(series) 也可取对数，但是这种情况后面所有模型（混合除外）均使用加法模型

    #画图
    def draw_ts(self):
        self.data.plot()
        plt.show()

    def adf_test(self):
        '''

        :return: ADF检验的P-value
        '''
        result = adfuller(self.data)
        return result[1]

    def decompose(self):
        '''
        对时序数据进行分解

        :return:输出趋势项、季节项、残差
        '''
        decomposition = seasonal_decompose(self.data)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        print(trend.dropna())
        print(seasonal.dropna())
        print(residual.dropna())
        return trend, seasonal, residual

    def regdetrend(self):
        '''
        用回归的方法去除趋势项

        :return: 返回回归后得到的去除了趋势项的数据
        '''
        X = [i for i in range(0, len(self.data))]
        X = np.reshape(X, (len(X), 1))
        y = self.data.values
        model = LinearRegression()
        model.fit(X, y)
        trend = model.predict(X)
        detrended = [y[i] - trend[i] for i in range(0, len(self.data))]
        return detrended

    def detrend(self,tendtype_list):
        '''
        基于decompose，一次性去除趋势性、季节性

        :param tendtype: 一个两个变量组成的list，第一个输入'add'或者'mul‘，代表加法/乘法模型
        第二个输入't+s'、't'、's'，代表有趋势项和季节项、仅有趋势项、仅有季节项
        :return: 返回去除趋势、季节项后的数据
        '''
        type=tendtype_list[0]
        tendtype=tendtype_list[1]
        if type=='add' and tendtype=='t+s':
            decomposition = seasonal_decompose(self.data, model="addcative")
            trend = decomposition.trend
            detrenddat = self.data - trend
            season = decomposition.seasonal
            dedata = detrenddat - season
        elif type=='mul' and tendtype=='t+s':
            decomposition = seasonal_decompose(self.data, model="multilicative")
            trend = decomposition.trend
            detrenddat = self.data/trend
            season = decomposition.seasonal
            dedata = detrenddat/ season
        elif type=='add' and tendtype=='t':
            decomposition = seasonal_decompose(self.data, model="addcative")
            trend = decomposition.trend
            dedata =( self.data)- trend
        elif type=='add' and tendtype=='s':
            decomposition = seasonal_decompose(self.data, model="addcative")
            season = decomposition.seasonal
            dedata = (self.data) - season
        elif type=='mul' and tendtype=='t':
            decomposition = seasonal_decompose(self.data, model="multilicative")
            trend = decomposition.trend
            dedata = (self.data) / trend
        else:
            decomposition = seasonal_decompose(self.data, model="multilicative")
            season = decomposition.seasonal
            dedata = (self.data)/ season
        return dedata

    def ma(self, n):
        '''
        用移动平均法+差分的方法消除季节性

        :param n: 移动平均阶数
        :return: 输出移动平均和差分后的数据，一般来说消除了季节项
        '''
        ma = self.data.rolling(n).mean()
        madiff=self.data-ma
        return madiff

    def expsmooth(self):
        '''
        用三次指数平滑消除趋势性和季节性

        :return: 返回用乘法模型、加法模型处理后的数据，一般来说去除了趋势和季节项
        '''
        adddata = (
            ExponentialSmoothing(self.data, seasonal_periods=4, trend='add', seasonal='add').fit(use_boxcox=True)).fittedvalues
        # 乘法模型
        muldata = (
            ExponentialSmoothing(self.data, seasonal_periods=4, trend='mul', seasonal='mul').fit(use_boxcox=True)).fittedvalues
        return adddata,muldata


path=os.getcwd()
data = pd.read_csv(path+'\\AirPassengers.csv')#换数据

#这个数据用取对数后做滑动平均差分比较合适
tstest=timeseries(data,'Month','#Passengers')
print(tstest.adf_test())#测试不显著
dedat=tstest.detrend(['add','t+s'])#
dedat.plot()
plt.show()
print(adfuller(dedat.dropna())[1])
