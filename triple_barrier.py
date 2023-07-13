import numpy as np
import math
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# -1 sell; 0 hold; 1 buy
class triple_barrier:
  '''
  example:
  ret = triple_barrier.triple_barrier(dataStock.Close, 1.1, 0.95, 10)
  StockTechRet=pd.concat([StockTech,ret],axis=1)
  '''
  def __init__(self):
    pass

  def triple_barrier(data, ub, lb, max_period):

      price = data.Close

      def end_price(s):
          return np.append(s[(s / s[0] > ub) | (s / s[0] < lb)], s[-1])[0]/s[0]

      r = np.array(range(max_period))

      def end_time(s):
          return np.append(r[(s / s[0] > ub) | (s / s[0] < lb)], max_period-1)[0]

      p = price.rolling(max_period).apply(end_price, raw=True).shift(-max_period+1)



      t = price.rolling(max_period).apply(end_time, raw=True).shift(-max_period+1)
      t = pd.Series([t.index[int(k+i)] if not math.isnan(k+i) else np.datetime64('NaT')
                    for i, k in enumerate(t)], index=t.index).dropna()

      signal = pd.Series(0, p.index)
      signal.loc[p > ub] = 1
      signal.loc[p < lb] = -1

      # ret = pd.DataFrame(dataStock.index)
      ret = pd.DataFrame(data.index)
      ret = ret.set_index('Date')
      ret['triple_barrier_profir'] = p
      ret['triple_barrier_sell_time'] = t
      ret['triple_barrier_signal'] = signal
      ret['triple_barrier_close'] = data.Close

      return ret
  
if __name__ == '__main__':
    StockTech = pd.read_csv("StockTech.csv")
    StockTech = StockTech.set_index('Date')
    StockTech.index = pd.DatetimeIndex(StockTech.index)
    datestamp = []
    for i in range(len(StockTech.index)):
        datestamp.append(int(datetime.datetime.timestamp(StockTech.index[i])))
    df_datestamp = pd.DataFrame({'datestamp': datestamp})
    StockTech['datestamp'] = datestamp
    StockTech_Nonan = StockTech.dropna()
    ret = triple_barrier.triple_barrier(StockTech_Nonan, 1.05, 0.85, 10)
    StockTechRet_Nonan = pd.concat([StockTech_Nonan,ret],axis=1)
    plt.figure('Stock Diagram',facecolor = 'lightgray')
    StockTechRet_Nonan.Close.plot()
    StockTechRet_Nonan.triple_barrier_signal.plot(secondary_y=True)
    plt.grid()
    plt.show()