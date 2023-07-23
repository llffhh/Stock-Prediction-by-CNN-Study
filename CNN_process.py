from keras import utils
import numpy as np
import tensorflow_addons as tfa



def process(train_x, train_y, test_x, test_y, cv_y, cv_x, model, batch_size, epochs, model_name):
    # Training progress
    ten_y=utils.to_categorical(train_y,3)
    train_history = model.fit(x=train_x,y=ten_y,batch_size=batch_size,epochs=epochs,shuffle=False)
    model.save(f'saved_model/{model_name}')

    # validation progress
    list_cv_y=utils.to_categorical(cv_y,3)

    list_cv_x=[]
    for i in range(len(cv_x)):
        list_cv_x.append(cv_x[i].tolist())
    list_cv_x=np.array(list_cv_x).reshape(len(list_cv_x),15,15,1)
    validation_loss, validation_acc = model.evaluate(list_cv_x,list_cv_y)
    print("Validation Loss = {}, Validation Acc. = {}".format(validation_loss,validation_acc))

    # Predict progress
    list_test_y=utils.to_categorical(test_y,3)  # 0, 1, -1

    list_test_x=[]
    for i in range(len(test_x)):
        list_test_x.append(test_x[i].tolist())
    list_test_x=np.array(list_test_x).reshape(len(test_x),15,15,1)
    pre_y = model.predict(list_test_x)
    # As pre_y have 3 column therefore there will have 0,1,2 after calculate as np.argmax.
    # Moreover, our label is 0, 1, -1. So we need to change the 2 to -1 when do the F1 score
    pre_y_label=np.argmax(pre_y,axis=1)

    # print("test y label:",np.reshape(test_y,(1,len(test_y))))
    for i in range(len(pre_y_label)):
        if pre_y_label[i]==2:
            pre_y_label[i]=-1
    # print("pred y label:",pre_y_label)

    # print(pre_y)
    metric = tfa.metrics.F1Score(num_classes=3)
    metric.update_state(list_test_y, pre_y)
    result = metric.result()
    test_loss, test_acc = model.evaluate(list_test_x,list_test_y)

    return pre_y_label, test_loss, test_acc, result, train_history