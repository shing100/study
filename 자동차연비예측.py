#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install -q seaborn')


# In[8]:


from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


# In[9]:


dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path


# In[10]:


column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()


# In[11]:


#데이터 정제하기, 이 데이터셋은 일부 데이터가 누락되어 있음
dataset.isna().sum()


# In[12]:


#누락된 행을 삭제
dataset = dataset.dropna()

# Origin 열은 수치형이 아니고 범주형이므로 원-핫 인코딩으로 변환
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japen'] = (origin == 3) * 1.0
dataset.tail()


# In[13]:


# 데이터셋을 훈련 세트와 테스트 세트로 분할라기
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# In[14]:


# 훈련 세트에서 몇 개의 열을 선택해 산점도 행렬
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")


# In[15]:


# 전반적인 통계 확인
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats


# In[16]:


# 전반적인 통계 확인
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


# In[17]:


# 데이터 정규화, train_stats 톨계를 다시 살펴보고 각 특성의 범위가 얼마나 다른지 확인.
# 특성의 스케일과 범위가 다르면 정규화하는 것이 권장, 특성을 정규화하지 않아도 모델이 수렴할 수 있지만,
# 훈련시키기 어렵고 입력 단위에 의존적인 모델이 만들어짐.
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# In[26]:


# 모델 만들기
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


# In[27]:


# 모델 확인
model = build_model()
model.summary()


# In[28]:


# 모델을 한번 실행.
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result


# In[34]:


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: 
            print('')
        print('.', end='')
        
EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split = 0.2, verbose = 0,
    callbacks=[PrintDot()]
)


# In[36]:


import matplotlib.pyplot as plt

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure(figsize=(8,12))
    
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    plt.ylim([0,5])
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()
             
plot_history(history)


# In[37]:


# 그래프를 보면 수백번 에포크를 진행한 이후에는 모델이 거의 향상되지 않음
# model.fit 메소드를 수정하여 검증 점수가 향상되지 않으면 자동으로 훈련을 멈추도록 변경
model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose = 0, callbacks=[early_stop, PrintDot()])

plot_history(history)


# In[38]:


# 모델을 훈련할 때 사용하지 않았던 테스트 세트에서 모델의 성능을 확인.
# 이를 통해 모델이 실전에 투입되었을 때 모델의 성능을 짐작할 수 있음
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))


# In[39]:


# MPG 값을 예측
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])


# In[40]:


# 오차분포
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")


# In[ ]:




