import numpy as np
import requests
import datetime
import json
import re
import pandas as pd
from lxml import etree



class StockCrawl():
    # set a default parameter
    def __init__(self,name):
        self.name = name+'.TW'
        self.url = 'https://query1.finance.yahoo.com/v7/finance/download/'
        self.start = '?period1='
        self.startdate = str(int(datetime.datetime(2000,1,1).timestamp()))
        self.end = '&period2='
        self.enddate = str(int(datetime.datetime.now().timestamp()))
        self.endhttps = '&interval=1d&events=history&includeAdjustedClose=true'
        self.headers = {'User-Agent':'Mozilla/5.0,Chrome/85.0.4183.102'} #Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36 Edg/85.0.564.51

    # set a date period
    def period(self, startdate=None):
        if startdate == None:
            startdate_re = self.startdate
        else:
            A = startdate.split('-',2)
            startdate_re = str(int(datetime.datetime(int(A[0]),int(A[1]),int(A[2])).timestamp()))
        return startdate_re

    # download the csv data file
    # https://query1.finance.yahoo.com/v7/finance/download/2330.TW?period1=1645285166&period2=1676821166&interval=1d&events=history&includeAdjustedClose=true  for download csv
    def csv_download(self, startdate=None):

        startdate_re = self.period(startdate)
        url_start = 'https://query1.finance.yahoo.com/v7/finance/download/'
        start_period = '?period1='
        end_period = '&period2='
        endhttps = '&interval=1d&events=history&includeAdjustedClose=true'
        url = url_start + self.name + start_period + startdate_re + end_period + self.enddate + endhttps

        # download start
        req = requests.get(url,headers = self.headers)
        open('/content/drive/MyDrive/test.csv', 'w').write(req.text)

        # modify into dataframe type
        req_text = req.text
        req_split = req_text.split("\n")
        data = [i.split(",") for i in req_split[1::]]
        data_columns = req_split[0].split(",")
        df = pd.DataFrame(data, columns=data_columns)
        df = df.set_index('Date')
        df.index = pd.DatetimeIndex(df.index)
        # change data type to float16 for Open, High, Low, Close; int32 for Volume and float16 for Adj Close
        df = df.astype({"Open": "float16", "High": "float16", "Low": "float16", "Close": "float16", "Adj Close": "float16", "Volume": "int32"})
        df = df.convert_dtypes()
        dfdn = df.interpolate(method='ffill')
        dfv = dfdn
        print(dfv.dtypes)
        return dfv

    # load page
    # https://finance.yahoo.com/quote/2330.TW/history?period1=946944000&period2=1602633600&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true'  for scraping
    def loadpage(self, startdate=None):

        startdate_re = self.period(startdate)
        startdate_re = self.period(startdate)
        url_start = 'https://finance.yahoo.com/quote/'
        start_period = '/history?period1='
        end_period = '&period2='
        endhttps = '&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true'
        url = url_start + self.name + start_period + startdate_re + end_period + self.enddate + endhttps

        # crawler start
        req = requests.get(url,headers = self.headers)
        dfv = self.parsepage(req)
        return dfv

    # parse page
    def parsepage(self,req):
        html = etree.HTML(req.content)
        patternDate = html.xpath("//td[@class='Py(10px) Ta(start) Pend(10px)']/span")
        patternOpen = html.xpath("//td[@class='Py(10px) Pstart(10px)'][1]/span")
        patternHigh = html.xpath("//td[@class='Py(10px) Pstart(10px)'][2]/span")
        patternLow = html.xpath("//td[@class='Py(10px) Pstart(10px)'][3]/span")
        patternClose = html.xpath("//td[@class='Py(10px) Pstart(10px)'][4]/span")
        patternVolume = html.xpath("//td[@class='Py(10px) Pstart(10px)'][6]/span")
        patternAdjclose = html.xpath("//td[@class='Py(10px) Pstart(10px)'][5]/span")

        dataDate = [i.text for i in patternDate]
        dataOpen = [i.text for i in patternOpen]
        dataHigh = [i.text for i in patternHigh]
        dataLow = [i.text for i in patternLow]
        dataClose = [i.text for i in patternClose]
        dataVolume_punc = [i.text for i in patternVolume]
        dataVolume = [i.replace(',','') for i in dataVolume_punc]
        dataAdjclose = [i.text for i in patternAdjclose]

        class dataDateformat():
            def __init__(self):
                self.dataDate_re=[]
                self.dataDateArr=[]
            def Dateformat(self,datalist):
                for i in range(len(datalist)):
                    if i == 0:
                        self.dataDateArr.append(datalist[i])
                    else:
                        if datalist[i] == datalist[i-1]:
                            pass
                        else:
                            self.dataDateArr.append(datalist[i])
                self.dataDate_re = self.dataDateArr
              return self.dataDate_re

        class datafloat():P
            def __init__(self):
                self.datafloat_re=[]
            def floating(self,datalist):
                for i in datalist:
                    if i == "null":
                        self.datafloat_re.append(np.NaN)
                    else:
                        self.datafloat_re.append(round(float(i),2))
                return self.datafloat_re

        dataDate_re = dataDateformat().Dateformat(dataDate)
        dataOpen_re = datafloat().floating(dataOpen)
        dataHigh_re = datafloat().floating(dataHigh)
        dataLow_re = datafloat().floating(dataLow)
        dataClose_re = datafloat().floating(dataClose)
        dataAdjclose_re = datafloat().floating(dataAdjclose)
        dataVolume_re = datafloat().floating(dataVolume)
        
        # dict type
        dataStock={'Date':dataDate_re,'open':dataOpen_re,'high':dataHigh_re,'low':dataLow_re,'close':dataClose_re,'Adj Close':dataAdjclose_re,
                  'volume':dataVolume_re}
        dfv = self.writepage(dataStock)
        return dfv

    # write page
    def writepage(self,dataStock):
        df = pd.DataFrame.from_dict(dataStock)
        df = df.sort_index(ascending = False)
        df = df.set_index('Date')
        df.index = pd.DatetimeIndex(df.index)
        dfdn = df.interpolate(method='ffill')
        dfv = dfdn.convert_dtypes()
        dfv.to_csv('/content/drive/MyDrive/'+self.name+'.csv')
        print(dfv)
        return dfv

if __name__ == '__main__':
    stock = "2330"
    StockCrawling = StockCrawl(stock)
    Stock_data = StockCrawling.csv_download(startdate="20180101")
    print(Stock_data)

    