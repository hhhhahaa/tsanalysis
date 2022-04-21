from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.parser import parse
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

path=os.getcwd()
data = pd.read_csv(path+'\\AirPassengers.csv')
data = data.set_index('Month')
data.index = pd.to_datetime(data.index)
ts = data.iloc[:,0]

#画图
def draw_ts(timeseries):
    timeseries.plot()
    plt.show()
import matplotlib.pyplot as plt

#ADF test，这个封不封装函数都一样，直接写也行，这里函数为了方便查看直接输出了pvalue，
def adf_test(data):
    result=adfuller(data)
    return result[1]

#分解。 输入数据为时序数据框
def decompose(data):
    decomposition = seasonal_decompose(data)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return trend, seasonal, residual

#去除趋势项
#差分
diffdt=ts.diff

#回归
def regdetrend(data):
    X = [i for i in range(0, len(data))]
    X = np.reshape(X, (len(X), 1))
    y = data.values
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict(X)
    detrended = [y[i] - trend[i] for i in range(0, len(data))]
    return detrended

#比值
def rat(data):
    ratio=data[1:]/data[:len(data)-1]-1
    return ratio

#消除季节项
#差分法同上，数据周期要调整好
#移动平均法，依旧会损失数据，n为滑动平均长度
def ma(data,n):
    ma=data.rolling(n).mean()
    return ma

#三次指数平滑
#加法模型
tsls=ts.values.tolist()
adddata= (ExponentialSmoothing(ts, seasonal_periods=4, trend='add', seasonal='add').fit(use_boxcox=True)).fittedvalues
#乘法模型
muldata =(ExponentialSmoothing(ts, seasonal_periods=4, trend='mul', seasonal='mul').fit(use_boxcox=True)).fittedvalues




