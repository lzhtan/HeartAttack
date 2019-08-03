
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn import metrics
import copy
import matplotlib.pyplot as plt
filepath = "/data/heartattack_enhance_data.csv"


# In[25]:


validPercentage = 0.1 #验证集占比
testPercentage = 0.1 #测试集占比

df = pd.read_csv(filepath)
data = df.values
n = data.shape[0]
numFeat = 7 #特征个数

Xo = data[:,1:8] #特征输入,1-7列
yi = data[:,10] #标签输出,第10列


# In[26]:


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


# In[27]:


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


rnd_clf = RandomForestRegressor(n_estimators=100, criterion='mse',max_leaf_nodes=16, n_jobs=-1, verbose=1)
rnd_clf.fit(X_train, y_train) #训练模型


# In[32]:


param_test1= {'n_estimators':range(1,101,1)}
gsearch1= GridSearchCV(estimator = RandomForestRegressor(criterion='mse',
                                                         max_leaf_nodes=16, n_jobs=-1, verbose=0),
                       param_grid =param_test1, scoring='neg_mean_squared_error',cv=2,verbose=2)  #对树的个数进行超参搜索
gsearch1.fit(X_valid,y_valid)


# In[33]:


print(gsearch1.best_params_)
print(gsearch1.best_score_)
#print(gsearch1.cv_results_)


# In[38]:


test_loss = gsearch1.cv_results_['mean_test_score'] # list，代表树的数目在test上的loss（其实不一定）
train_loss = gsearch1.cv_results_['mean_train_score'] # list，代表树的数目在train上的loss（其实不一定）
test_loss = np.asarray(test_loss) #为了方便取反，转成ndarray
train_loss = np.asarray(train_loss) #
test_loss = -test_loss #这个loss是负的，为了画图，把它取反
train_loss = -train_loss #

#画图展示一下预测值和真实值
nums = range(0,100)
plt.figure(figsize=(20,5))
plt.plot(nums, smooth_curve(train_loss,0), 'r--', label='train_loss')
plt.plot(nums, smooth_curve(test_loss,0), 'b', label='test_loss')
plt.title('predict value and true value')
plt.legend()
plt.show()

print(list(np.round(train_loss,6)))
print(list(np.round(test_loss,6)))


# In[31]:


result = rnd_clf.predict(X_test) #预测模型
result2 = np.where(result>0.5, 1, 0) #将概率二值化
y_test = [int(i) for i in list(y_test)] #将1.0 和0.0 变为 1和0
print(list(np.round(result,2)))
print(list(result2))
print(y_test)

