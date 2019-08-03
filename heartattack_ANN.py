
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import copy
import matplotlib.pyplot as plt
filepath = "/data/heartattack_enhance_data.csv"


# In[2]:


validPercentage = 0.1 #验证集占比
testPercentage = 0.1 #测试集占比

df = pd.read_csv(filepath)
n = data.shape[0] #数据样本数
data = df.values
numFeat = 7 #特征个数

Xo = data[:,1:8] #特征输入,1-7列
yi = data[:,10] #标签输出,第10列


# In[5]:


#查看各列数据缺失情况
for index in df.columns.values:
    print(df[index].isnull().value_counts())


# In[7]:


# Generate Dataset
"""
划分成3个样本集，训练集Training，验证集Validation，测试集Test
"""
lastTraining = int(n*(1-validPercentage-testPercentage))
lastValid = int(n*(1-testPercentage))

X_train = Xo[0:lastTraining,:] #0-lastTraining行，所有列
y_train = yi[0:lastTraining] #0-lastTraining行，所有列

X_valid = Xo[lastTraining:lastValid,:] #lastTraining-lastValid行，所有列
y_valid = yi[lastTraining:lastValid] #lastTraining-lastValid行，所有列

X_test = Xo[lastValid:n,:] #lastValid-n行，所有列
y_test = yi[lastValid:n] #lastValid-n行，所有列


# In[8]:


#k1作为metrics
from keras import backend as K

def f1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) #1代表正类
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

##数据曲线平滑化##
def smooth_curve(points, factor=0.9): #利用指数加权平均数来平滑曲线
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# In[28]:


##建立模型##
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(20, activation='relu', input_shape=(numFeat,)))
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
          optimizer= "Adam",
          metrics=[f1_score])
    return model


# In[31]:


##开始训练##
model = build_model()
history = model.fit(X_train, y_train, epochs=500, batch_size=100, validation_data=(X_valid, y_valid))


# In[63]:


#画图展示一下训练过程
import matplotlib.pyplot as plt
mae = history.history['f1_score'] 
val_mae = history.history['val_f1_score'] 
loss = history.history['loss'] 
val_loss = history.history['val_loss']

epochs = range(1, len(mae) + 1)

plt.plot(epochs, mae, 'r--', label='Training f1_score') 
plt.plot(epochs, val_mae, 'b', label='Validation f1_score') 
plt.title('Training and validation f1_score')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'r--', label='Training loss') 
plt.plot(epochs, val_loss, 'b', label='Validation loss') 
plt.title('Training and validation loss') 
plt.legend()

plt.show()


# In[40]:


result = model.predict(X_test)
result = result[:,0] #取所有行，第一列
result = np.around(result, decimals=2, out=None) #限制小数位数
result2 = np.where(result>0.5, 1, 0) #将概率二值化
#画图展示一下预测值和真实值
nums = range(0,763)
plt.figure(figsize=(20,5)) 
plt.plot(nums, result2, 'r--', label='predict value') 
plt.plot(nums, y_test, 'b', label='true value') 
plt.title('predict value and true value')

plt.legend()
plt.show()


# In[42]:


y_test = [int(i) for i in list(y_test)] #将1.0 和0.0 变为 1和0
print(list(result))
print(list(result2))
print(y_test)


# In[64]:


loss = np.round(loss,6)
val_loss =  np.round(val_loss,6)
print(list(loss))
print(list(val_loss))

