import numpy as np
import pandas as pd
'''由heartattack_data得到heartattack_enhance_data'''

from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import copy
import matplotlib.pyplot as plt
filepath = "/data/heartattack_data.csv"

#读取数据
df = pd.read_csv(filepath)
df = df.sample(frac=1) # random shuffle rows
df['age']=df['age'].fillna(df['age'].mean())
df = df.fillna(0)

df['infect'] = df['infect'].replace(2,0)
df['fever'] = df['fever'].replace(2,0)

#数据处理
df_fever = df[df['fever']==1]
print(df_fever.columns) # pandas.core.indexes.mumeric.Int64Index类
fever_df = {}
for col in df_fever.columns: #加入一些数据，并让其缺失一部分
    if col == 'id':
        continue
    fever_df[col] = copy.deepcopy(df_fever)
    fever_df[col][col] = None
    df = df.append(fever_df[col])
    if col == 'antibiotic_company':
        break
        
df.drop_duplicates(inplace=True) #去除重复值
df = df.sample(frac=1) # random shuffle rows
df = df.fillna(-1) #缺失值变为-1，代表未知

df.to_csv('heartattack_enhance_data.csv',index=False)