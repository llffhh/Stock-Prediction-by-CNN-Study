from Technical_Index import TechIndicator
from triple_barrier import triple_barrier
from financial_calculation import FinancialTesting
from StockCrawl import StockCrawl
from confusion_matrix import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from CNN_model import Model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from CNN_process import process


# Preprocessing
# download stock information
stock = "1504"
StockCrawling = StockCrawl(stock)
dataStock = StockCrawling.csv_download(startdate="2018-01-01")


# Calculate the techincal indicator
PdFrame = pd.DataFrame(dataStock.index)
PdFrame = PdFrame.set_index('Date')

for i in range(6,21):
    print("period: {}".format(i))
    PdFrame[f"RSI_{i}"]=TechIndicator().RSI(dataStock,i)
    PdFrame[f"WilliamR_{i}"]=TechIndicator().WilliamR(dataStock,i)
    PdFrame[f"WMA_{i}"]=TechIndicator().WMA(dataStock,i)
    PdFrame[f"EMA_{i}"]=TechIndicator().EMA(dataStock,i)
    PdFrame[f"MA_{i}"]=TechIndicator().MA(dataStock,i)
    PdFrame[f"HMA_{i}"]=TechIndicator().HMA(dataStock,i)
    PdFrame[f"TEMA_{i}"]=TechIndicator().TEMA(dataStock,i)
    PdFrame[f"CCI_{i}"]=TechIndicator().CCI(dataStock,i)
    PdFrame[f"CMO_{i}"]=TechIndicator().CMO(dataStock,i)
    PdFrame[f"MACD_{i}"]=TechIndicator().MACD(dataStock,i)[2]
    PdFrame[f"PPO_{i}"]=TechIndicator().PPO(dataStock,i)
    PdFrame[f"ROC_{i}"]=TechIndicator().ROC(dataStock,i)
    PdFrame[f"CMFI_{i}"]=TechIndicator().CMFI(dataStock,i)
    PdFrame[f"DMI_{i}"]=TechIndicator().DMI(dataStock,i)[2]
    PdFrame[f"SAR_{i}"]=TechIndicator().SAR(dataStock,i)



StockTech = pd.concat([dataStock,PdFrame],axis = 1)

# df = pd.read_csv("2330.TW.csv")
# df = df.dropna()
# df = df.set_index('Date')
# df.index = pd.DatetimeIndex(df.index)
# dataStock = df
# StockTech = pd.read_csv("StockTech.csv")
# StockTech = StockTech.set_index('Date')
# StockTech.index = pd.DatetimeIndex(StockTech.index)

#Filter the all the Nan data or select the date we need
datestamp = []
for i in range(len(StockTech.index)):
    datestamp.append(int(datetime.datetime.timestamp(StockTech.index[i])))
df_datestamp = pd.DataFrame({'datestamp': datestamp})
StockTech['datestamp'] = datestamp
StockTech_Nonan = StockTech.dropna()
print(StockTech_Nonan)

#label the data and take out the Technical indicator and the label
ret = triple_barrier.triple_barrier(StockTech_Nonan, 1.1, 0.9, 10)
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
FinancialTesting(StockTechRet_Nonan, 5, 10000, 10, 5, 1).process()




# Machine learning algorithm
# Postprocessing
# CNN_model net
Model_net = Model.net()

result_net = process(train_x, train_y, test_x, test_y, cv_y, cv_x, Model_net, batch_size=1028, epochs=100)

# CNN_model net1
lr=1e-3
total_steps=10000
warmup_proportion=0.1
min_lr=1e-5
Model_net1 = Model.net1(lr,total_steps,warmup_proportion,min_lr)

result_net1 = process(train_x, train_y, test_x, test_y, cv_y, cv_x, Model_net1, batch_size=2000, epochs=800)

# gather results
Model_result = [[result_net, Model_net], [result_net1,Model_net1]]

# print results
for i in Model_result:
    result_data, model = i
    pre_y_label, test_loss, test_acc, result, train_history = result_data
    # Print Model
    print("============================================================")
    print(model.summary())

    # print F1Score
    print("Test Loss = {}, Test Acc. = {}".format(test_loss,test_acc))
    print("F1Score --> 0: {}, 1: {}, -1: {}".format(result.numpy()[0],result.numpy()[1],result.numpy()[2]))

    ConfusionMatrix = confusion_matrix(test_y,pre_y_label)
    print(classification_report(test_y, pre_y_label))

    losses = list(train_history.history['loss'])
    accuracies = list(train_history.history['acc'])
    plt.figure(facecolor='lightgray')
    plt.subplot(1,2,1)
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');
    plt.subplot(1,2,2)
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy vs. No. of epochs');

    # draw the signal and pred_y_label
    plt.figure(facecolor='lightgray')
    start = len(StockTechRet_Nonan)-len(test_x)
    end = len(StockTechRet_Nonan)
    price = StockTechRet_Nonan.Close[start:end]
    date = StockTechRet_Nonan.index[start:end]
    signal = StockTechRet_Nonan.triple_barrier_signal[start:end]
    pred_y = pre_y_label
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(date,price,label='price')
    ax1 = ax.twinx()
    ax1.plot(date,signal,'r',marker = 'o',label = "signal")
    ax1.plot(date,pred_y,'b',label = 'pred_y')
    ax.legend()
    ax1.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Closed Price')
    ax1.set_ylabel('Signal')
    # ax.xaxis.set_major_locator(mdates.MonthLocator(date))
    plt.grid()

    # Plot the Confusion Matrix
    plot_confusion_matrix(ConfusionMatrix, normalize = False, target_names = ['stop loss', 'hold', 'stop profit'], title = "Confusion Matrix")


    # Calculate the Financial Result
    print("Original Signal base on Triple Barrier Method")
    FinancialTesting(StockTechRet_Nonan[start:end],10,1000,1,10,1).process()
    Pred_DataFrame = pd.DataFrame(StockTechRet_Nonan.index[start:end])
    Pred_DataFrame = Pred_DataFrame.set_index('Date')
    Pred_DataFrame['Close'] = StockTechRet_Nonan.Close[start:end]
    Pred_DataFrame['triple_barrier_sell_time'] = StockTechRet_Nonan.triple_barrier_sell_time[start:end]
    Pred_DataFrame['triple_barrier_signal'] = pre_y_label
    print("\nPrediction Signal by CNN")
    FinancialTesting(Pred_DataFrame,10,1000,1,10,1).process()