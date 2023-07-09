import numpy as np
import pandas as pd

class TechIndicator:
  def __init__(self):
    pass

  def WilderSmooth(self,data,day):
    tday = day-1
    AveWSGain = np.zeros(len(data))
    AveWSLoss = np.zeros(len(data))
    Gain = np.zeros(len(data))
    Loss = np.zeros(len(data))
    for i in range(len(data)-1):
      if data.Close[i+1]>=data.Close[i]:
        Gain[i+1] = data.Close[i+1]-data.Close[i]
      else:
        Loss[i+1] = data.Close[i+1]-data.Close[i]
    for i in range(len(data)-tday):
      if i == 0:
        AveWSGain[tday+i] = np.average(Gain[i:tday+1])
        AveWSLoss[tday+i] = np.average(Loss[i:tday+1])
      else:
        AveWSGain[tday+i] = AveWSGain[tday+i-1]+(Gain[tday+i]-AveWSGain[tday+i-1])/(tday+1)
        AveWSLoss[tday+i] = AveWSLoss[tday+i-1]+(Loss[tday+i]-AveWSLoss[tday+i-1])/(tday+1)
    return AveWSGain, AveWSLoss, Gain, Loss


  def RSI(self,data,day):
    #rsi = np.zeros(len(data))
    rsi = np.full(len(data),np.nan)
    AveWSGain,AveWSLoss, Gain, Loss = self.WilderSmooth(data,day)
    RS = AveWSGain/np.abs(AveWSLoss)
    rsi = 100*(RS/(1+RS))
    return rsi


  def WilliamR(self,data,day):
    tday = day-1
    #WR = np.zeros(len(data))
    WR = np.full(len(data),np.nan)
    for i in range(len(data)-tday):
      WR[tday+i] = ((np.max(data.High[i:tday+i+1])-data.Close[tday+i])
      /(np.max(data.High[i:tday+i+1])-np.min(data.Low[i:tday+i+1])))*(-100)
    return WR

  def MA(self,data,day):
    tday = day-1
    #MAt = np.zeros(len(data))
    MAt = np.full(len(data),np.nan)
    for i in range(len(data)-tday):
      MAt[tday+i] = np.average(data.Close[i:tday+i+1])
    return MAt

  def EMA(self,data,day):
    tday = day - 1
    EMAt = np.zeros(len(data))
    #EMAt = np.full(len(data),np.nan)
    for i in range(len(data)-tday):
      if i == 0:
        EMAt[tday+i] = np.average(data.Close[i:tday+i+1])
      else:
        EMAt[tday+i] = EMAt[tday+i-1]+2/(tday+1+1)*(data.Close[tday+i]-EMAt[tday+i-1])
    #EMAt[0:tday] = np.nan
    return EMAt

  def WMA(self,data,day):
    tday = day - 1
    #WMAt = np.zeros(len(data))
    WMAt = np.full(len(data),np.nan)
    Wn = np.array(list(range(1,tday+2)))
    for i in range(len(data)-tday):
      WMAt[tday+i] = np.average(np.array(data.Close[i:tday+i+1]),weights=Wn)
    return WMAt

  def HMA(self,data,day):
    #HMAt = np.zeros(len(data))
    HMAt = np.full(len(data),np.nan)
    #產生由完整時間區域所產生的HMA
    HMA_f = self.WMA(data,day)
    #產生由一半的時間區域所產生的HMA
    tday = int(np.ceil(day/2))+1
    HMA_h = self.WMA(data,tday)
    HMA_hf = 2*HMA_h - HMA_f
    tday = int(np.ceil(np.sqrt(day)))+1
    data = pd.DataFrame.from_dict({'Close':HMA_hf})
    HMAt = self.WMA(data,tday)
    return HMAt

  def TEMA(self,data,day):
    #first time of EMA
    EMAt=self.EMA(data,day)
    data=pd.DataFrame.from_dict({'Close':EMAt})
    #second time of EMA
    EMAtt=self.EMA(data,day)
    data=pd.DataFrame.from_dict({'Close':EMAtt})
    #third time of EMA
    EMAttt=self.EMA(data,day)
    TEMA = 3*EMAt-3*EMAtt+EMAttt
    #TEMA[0:day]=0
    TEMA[0:(day-1)*3]=[np.nan]
    return TEMA

  def CCI(self,data,day):
    '''
    TP = (close + high + low)/3
    MA = 20 period SMA of TP
    CCI = (TP-MA)/(0.015*mean deviation)
    '''
    tday = day-1
    TP = (data.Close+data.High+data.Low)/3
    MA = np.zeros(len(data))
    #MA = np.full(len(data),np.nan)
    MD = np.zeros(len(data))
    #MD = np.full(len(data),np.nan)
    for i in range(len(TP)-tday):
      MDt = [] # initialize the Mean Deviation
      MAt = np.average(TP[i:day+i])
      for j in range(day):
        MDt.append(abs(TP[i+j]-MAt))
      MD[tday+i] = np.average(MDt)
      MA[tday+i] = MAt
    CCI = (TP-MA)/(0.015*MD)

    CCI[0:tday]=np.nan
    # print(len(CCI))
    return CCI

  def CMO(self,data,day):
    tday = day - 1
    AveWSGain,AveWSLoss, Gain, Loss = self.WilderSmooth(data,day)
    #CMOt = np.zeros(len(data))
    CMOt = np.full(len(data),np.nan)
    for i in range(len(data)-tday):
      CMOt[tday+i] = (np.sum(Gain[i:day+i])-abs(np.sum(Loss[i:day+i])))/(np.sum(Gain[i:day+i])+abs(np.sum(Loss[i:day+i])))*100
    return CMOt

  def MACD(self,data,day):
    EMA12 = self.EMA(data,12)
    EMA26 = self.EMA(data,26)
    EMA12[0:25] = [0]
    MACD = EMA12 - EMA26
    signalLine = self.EMA(pd.DataFrame.from_dict({'Close':MACD}),9)
    MACDhist = MACD-signalLine
    MACDhist[0:33] = [np.nan]
    return MACD,signalLine,MACDhist

  def PPO(self,data,day): #論文用MA
    PPO = (self.MA(data,12)-self.MA(data,26))/self.MA(data,26)*100
    signalLine = self.EMA(pd.DataFrame.from_dict({'Close':PPO}),9)
    return PPO

  def ROC(self,data,day):
    tday = day - 1
    #ROC = np.zeros(len(data))
    ROC = np.full(len(data),np.nan)
    for i in range(len(data)-tday):
      ROC[tday+i] = (data.Close[i+tday]-data.Close[i])/data.Close[i]*100
    return ROC

  def CMFI(self,data,day):    # 12 period
    tday = 12 - 1
    Multiplier = ((data.Close-data.Low)-(data.High-data.Close))/(data.High-data.Low)
    MFV = data.Volume*Multiplier  # Money Flow Volume
    #CMFI = np.zeros(len(data))
    CMFI = np.full(len(data),np.nan)
    for i in range(len(data)-tday):
      CMFI[tday+i] = np.sum(Multiplier[i:tday+i+1])/np.sum(MFV[i:tday+i+1])
    return CMFI

  def DMI(self,data,day):  # still got some problem
    tday = day - 1
    UpMove = np.zeros(len(data))
    DownMove = np.zeros(len(data))
    posDMI = np.zeros(len(data))
    negDMI = np.zeros(len(data))
    TR = np.zeros(len(data))
    #TR = np.full(len(data),np.nan)
    tt = 1
    for i in range(len(data)-tt):
      UpMove[tt+i] = data.High[i+tt] - data.High[i]
      DownMove[tt+i] = data.Low[i] - data.Low[i+tt]
      if UpMove[tt+i]>DownMove[tt+i] and UpMove[tt+i]>0:
        posDMI[tt+i] = UpMove[tt+i]
      else:
        posDMI[tt+i] = 0

      if DownMove[tt+i]>UpMove[tt+i] and DownMove[tt+i]>0:
        negDMI[tt+i] = DownMove[tt+i]
      else:
        negDMI[tt+i] = 0

      if posDMI[tt+i]>negDMI[tt+1]:
        negDMI[tt+i] = 0
      else:
        posDMI[tt+i] = 0
      TRprice = [data.High[tt+i]-data.Low[tt+i], data.High[tt+i]-data.Close[i], data.Low[tt+i]-data.Close[i]]
      TR[tt+i] = np.max(np.abs(TRprice))

    posDI = 100*self.EMA(pd.DataFrame.from_dict({'Close':posDMI}),day)/self.EMA(pd.DataFrame.from_dict({'Close':TR}),day)
    negDI = 100*self.EMA(pd.DataFrame.from_dict({'Close':negDMI}),day)/self.EMA(pd.DataFrame.from_dict({'Close':TR}),day)
    DX = abs((posDI-negDI)/(posDI+negDI))*100
    DX[0:tday] = 0
    ADX = self.EMA(pd.DataFrame.from_dict({'Close':DX}),day)
    ADX[0:tday] = np.nan
    return posDI,negDI,ADX,DX,posDMI,negDMI

  def SAR(self,data,day):
    def RPSAR(PSAR,AF,EP):
      RPSAR = PSAR + AF*(EP-PSAR)
      return RPSAR
    def FPSAR(PSAR,AF,EP):
      FPSAR = PSAR - AF*(PSAR-EP)
      return FPSAR

    [AF,AF_max] = [0.02,0.2]
    #SAR = np.zeros(len(data))
    SAR = np.full(len(data),np.nan)
    period = 5
    tt = period - 1
    for i in range(len(data)-tt):
      if i == 0:
        if dataStock.Close[i+tt]>dataStock.Close[i]: #uptrend
          PSAR = np.min(dataStock.Low[i:i+period])
          EP = np.max(dataStock.High[i:i+period])
        else:
          PSAR = np.max(dataStock.High[i:i+period])
          EP = np.min(dataStock.Low[i:i+period])
      else:
        if dataStock.Close[i+tt]>PSAR:       #uptrend
          PSAR = RPSAR(PSAR,AF,EP)
          if dataStock.Close[i+tt-1]<PSAR:
            AF = 0.02
          if dataStock.High[i+tt]>EP:
            if AF >= AF_max:
              AF = 0.2
            else:
              AF += 0.04
            EP = dataStock.High[i+tt]
        else:                               #downtrend
          PSAR = FPSAR(PSAR,AF,EP)
          if dataStock.Close[i+tt-1]>PSAR:
            AF = 0.02
          if dataStock.Low[i+tt]<EP:
            if AF >= AF_max:
              AF = 0.2
            else:
              AF += 0.04
            EP = dataStock.Low[i+tt]
      SAR[i+tt] = PSAR
    return SAR
  
