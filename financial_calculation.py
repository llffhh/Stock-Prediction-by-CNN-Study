class FinancialTesting:
  '''
  FinancialTesting(data, period, cash, eachStock, triple_barrier_period, passlosssignal)
  period = How long do we buy the stock
  cash = Initial Cash
  eachStock = How many stock do we buy per time
  triple_barrier_period = The period we set in triple barrier function
  passlosssignal = if 1, mean we don't buy any stock when the signal is stop loss signal
  '''
  def __init__(self, data, period, cash, eachStock, triple_barrier_period, passlosssignal):
    self.period = period
    self.stockNumber = 0
    self.StockValue = 0
    self.Cash = cash
    self.EachTime = eachStock # 100 stock
    self.signalRecord = []
    self.CashList = []
    self.StockCost = 0
    self.StockTechRet_Nonan = data
    self.passlosssignal = passlosssignal
    self.triple_barrier_period = triple_barrier_period
    self.buying_price = []
    self.selling_price = []

  def buying(self, signal, stockcost_eachTime):
    # print("Buying signal is triggered ({} period), signal: {}".format(self.period, signal))
    self.Cash -= stockcost_eachTime
    self.stockNumber += self.EachTime
    self.StockCost += stockcost_eachTime
    self.signalRecord.append(signal)
    # print("Buying price: {}, Cash: {}".format(self.StockTechRet_Nonan.Close[i],self.Cash))

  def sellingsignal(self, signal, stockcost_eachTime, futureTime):
    difference = self.StockTechRet_Nonan.Close[futureTime]*self.EachTime-stockcost_eachTime
    self.Cash += difference + stockcost_eachTime
    self.StockCost -= stockcost_eachTime
    self.stockNumber -= self.EachTime
    return difference

  def process(self):
    for i in range(len(self.StockTechRet_Nonan)-self.triple_barrier_period):
      futureTime = self.StockTechRet_Nonan.triple_barrier_sell_time[i]
      signal = self.StockTechRet_Nonan.triple_barrier_signal[i]
      # Buy the Stock after a specific period and calculate the profit we make after specific period
      if i == 0 or (i+1)%self.period==0:
        if signal == 1: #stop profit
          stockcost_eachTime = self.StockTechRet_Nonan.Close[i]*self.EachTime
          self.buying(signal, stockcost_eachTime)
          difference = self.sellingsignal(signal, stockcost_eachTime, futureTime)
          self.buying_price.append(self.StockTechRet_Nonan.Close[i])
          self.selling_price.append(self.StockTechRet_Nonan.Close[futureTime])
          # print('Selling price: {}, Cash: {}, Stop Profit: {}\n'.format(self.StockTechRet_Nonan.Close[futureTime], self.Cash, difference))
        elif signal == -1 and self.passlosssignal == 0: #stop loss
          stockcost_eachTime = self.StockTechRet_Nonan.Close[i]*self.EachTime
          self.buying(signal, stockcost_eachTime)
          difference = self.sellingsignal(signal, stockcost_eachTime, futureTime)
          self.buying_price.append(self.StockTechRet_Nonan.Close[i])
          self.selling_price.append(self.StockTechRet_Nonan.Close[futureTime])
          # print('Selling price: {}, Cash: {}, Stop Loss: {}\n'.format(self.StockTechRet_Nonan.Close[futureTime], self.Cash, difference))
        elif signal == 0:
          stockcost_eachTime = self.StockTechRet_Nonan.Close[i]*self.EachTime
          self.buying(signal, stockcost_eachTime)
          self.buying_price.append(self.StockTechRet_Nonan.Close[i])
          # difference = self.sellingsignal(signal, stockcost_eachTime, futureTime)

      # Calculate the profit\loss we make when hold signal is triggered, stock number equal to each stock
      if self.stockNumber > 0:
        if self.StockTechRet_Nonan.triple_barrier_signal[i] == 1:
          stockcost_eachTime = self.StockCost/self.stockNumber*self.EachTime
          difference = self.sellingsignal(signal, stockcost_eachTime, futureTime)
          self.selling_price.append(self.StockTechRet_Nonan.Close[futureTime])
          # print('Selling price: {}, Cash: {}, Stop Profit: {}\n'.format(self.StockTechRet_Nonan.Close[futureTime], self.Cash, difference))
        elif self.StockTechRet_Nonan.triple_barrier_signal[i] == -1:
          stockcost_eachTime = self.StockCost/self.stockNumber*self.EachTime
          difference = self.sellingsignal(signal, stockcost_eachTime, futureTime)
          self.selling_price.append(self.StockTechRet_Nonan.Close[futureTime])
          # print('Selling price: {}, Cash: {}, Stop Loss: {}\n'.format(self.StockTechRet_Nonan.Close[futureTime], self.Cash, difference))

      self.CashList.append(self.Cash)
      self.StockValue = self.stockNumber * self.StockTechRet_Nonan.Close[i]

    print('The buying signal in these days: {}'.format(self.signalRecord))
    print('Buying price: {}'.format(self.buying_price))
    print('Selling price: {}'.format(self.selling_price))
    # print("Count -1: {}; Count 0: {}; Count 1: {}".format(self.signalRecord.count(-1), self.signalRecord.count(0), self.signalRecord.count(1)))

    print('Cash: {}'.format(self.Cash))
    # print('As we still have {} stock, therefore the Stock Value now is {}'.format(self.stockNumber, self.StockValue))

if __name__ == '__main__':
  FinancialTesting(StockTechRet_Nonan, 20, 10000, 1, 20, 1).process()