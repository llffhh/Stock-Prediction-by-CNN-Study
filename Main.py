from Technical_Index import TechIndicator
from triple_barrier import triple_barrier
from financial_calculation import FinancialTesting
from StockCrawl import StockCrawl
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import os


# Preprocessing
#download stock information
# stock = "2330"
# StockCrawling = StockCrawl(stock)
# dataStock = StockCrawling.csv_download(startdate="2018-01-01")
# df = pd.read_csv("2330.TW.csv")
# df = df.dropna()
# df = df.set_index('Date')
# df.index = pd.DatetimeIndex(df.index)
# dataStock = df

#Calculate the techincal indicator
# PdFrame = pd.DataFrame(dataStock.index)
# PdFrame = PdFrame.set_index('Date')

# for i in range(6,21):
    # print("period: {}".format(i))
    # PdFrame[f"RSI_{i}"]=TechIndicator().RSI(dataStock,i)
    # PdFrame[f"WilliamR_{i}"]=TechIndicator().WilliamR(dataStock,i)
    # PdFrame[f"WMA_{i}"]=TechIndicator().WMA(dataStock,i)
    # PdFrame[f"EMA_{i}"]=TechIndicator().EMA(dataStock,i)
    # PdFrame[f"MA_{i}"]=TechIndicator().MA(dataStock,i)
    # PdFrame[f"HMA_{i}"]=TechIndicator().HMA(dataStock,i)
    # PdFrame[f"TEMA_{i}"]=TechIndicator().TEMA(dataStock,i)
    # PdFrame[f"CCI_{i}"]=TechIndicator().CCI(dataStock,i)
    # PdFrame[f"CMO_{i}"]=TechIndicator().CMO(dataStock,i)
    # PdFrame[f"MACD_{i}"]=TechIndicator().MACD(dataStock,i)[2]
    # PdFrame[f"PPO_{i}"]=TechIndicator().PPO(dataStock,i)
    # PdFrame[f"ROC_{i}"]=TechIndicator().ROC(dataStock,i)
    # PdFrame[f"CMFI_{i}"]=TechIndicator().CMFI(dataStock,i)
    # PdFrame[f"DMI_{i}"]=TechIndicator().DMI(dataStock,i)[2]
    # PdFrame[f"SAR_{i}"]=TechIndicator().SAR(dataStock,i)



# StockTech = pd.concat([dataStock,PdFrame],axis = 1)
StockTech = pd.read_csv("StockTech.csv")
StockTech = StockTech.set_index('Date')
StockTech.index = pd.DatetimeIndex(StockTech.index)

#Filter the all the Nan data or select the date we need
datestamp = []
for i in range(len(StockTech.index)):
    datestamp.append(int(datetime.datetime.timestamp(StockTech.index[i])))
df_datestamp = pd.DataFrame({'datestamp': datestamp})
StockTech['datestamp'] = datestamp
StockTech_Nonan = StockTech.dropna()
print(StockTech_Nonan)

#label the data and take out the Technical indicator and the label
ret = triple_barrier.triple_barrier(StockTech_Nonan, 1.2, 0.8, 10)
StockTechRet_Nonan = pd.concat([StockTech_Nonan,ret],axis=1)
StockTechRet_Nonan_TechInd = StockTechRet_Nonan.iloc[:len(StockTechRet_Nonan),6:15*15+6]


#Nomarlized (Min-Max Scaled)
scaler = MinMaxScaler(feature_range=(-1,1))
StockTechRet_Nonan_TechInd_Nom = np.array(StockTechRet_Nonan_TechInd)


#change to pic size and input type
img_StockTechRet_Nonan_TechInd_Nom = []
for i in range(len(StockTechRet_Nonan_TechInd_Nom)):
    img_StockTechRet_Nonan_TechInd_Nom.append(np.transpose(scaler.fit_transform(StockTechRet_Nonan_TechInd_Nom[i].reshape((15,15)))).tolist())
StockTechRet_Nonan['img'] = img_StockTechRet_Nonan_TechInd_Nom
listData = []
listLabel = []
for i in range(len(StockTechRet_Nonan_TechInd_Nom)):
    listData.append(StockTechRet_Nonan.img.values[i])
    listLabel.append(StockTechRet_Nonan.triple_barrier_signal.values[i])
listData = np.array(listData).reshape(len(StockTechRet_Nonan_TechInd_Nom),15,15,1)
listLabel = np.array(listLabel).reshape(len(StockTechRet_Nonan_TechInd_Nom),1)
print("change to img type")

# train set and test set, 20天訓練，10天測試
data_StockTechRet = StockTechRet_Nonan['img'].values
label_StockTechRet = StockTechRet_Nonan['triple_barrier_signal'].values
train_x, test_x, train_y, test_y = train_test_split(listData, listLabel, train_size=0.8, test_size=0.2, random_state=2, shuffle=False)
train_x, cv_x, train_y, cv_y = train_test_split(train_x, train_y, train_size=0.8, test_size=0.2, random_state=2, shuffle=False)

# output pic example by randomly selecting
picNumber = 8
images_labels = list(zip(StockTechRet_Nonan.img.values,StockTechRet_Nonan.triple_barrier_signal.values))
Random_img = np.random.randint(len(images_labels),size=picNumber)
for i, num in enumerate(Random_img):
    plt.figure('Img & Signal Diagram',facecolor = 'lightgray')
    plt.subplot(2,4,i+1)
    plt.axis('off')
    plt.imshow(images_labels[num][0], cmap = 'gray')
    plt.title('L: '+ str(images_labels[num][1]))
plt.show()

# calculate the number of hold, buy and sell
_labels, _counts = np.unique(train_y, return_counts = True)
for label, count in zip(_labels, _counts):
    # print(f"Percentage of class -1 = {_counts[0]/len(train_y)*100}, class 0 = {_counts[1]/len(train_y)*100}, class 1 = {_counts[2]/len(train_y)*100}")
    print(f"Percentage of class {label} = {count/len(train_y)*100}")

# plot the final data curve
plt.figure('Stock Diagram',facecolor = 'lightgray')
StockTechRet_Nonan.Close.plot()
StockTechRet_Nonan.triple_barrier_signal.plot(secondary_y=True)
plt.grid()
plt.show()

# FinancialTesting(data, period, cash, eachStock, triple_barrier_period, passlosssignal)
FinancialTesting(StockTechRet_Nonan, 10, 10000, 10, 10, 1).process()

# os.system('pause')