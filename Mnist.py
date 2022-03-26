#!/usr/bin/env python
# coding: utf-8

# In[5]:


from tensorflow import keras


# In[38]:


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# In[39]:


print(x_train.shape, x_test.shape)


# In[40]:


print(y_train.shape, y_test.shape)


# In[41]:


for i in range(3):
    print(y_train[i])


# In[42]:


y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


# In[43]:


for i in range(3):
    print(y_train[i])


# In[44]:


print(x_train.shape, x_test.shape)


# In[45]:


x_train = x_train.reshape([60000, 28*28])
x_test = x_test.reshape([10000, 28*28])


# In[46]:


print(x_train.shape, x_test.shape)


# In[72]:


model = keras.Sequential()
model.add(keras.layers.Dense(128, activation="sigmoid", input_shape=(28*28,)))
model.add(keras.layers.Dense(128, activation="sigmoid"))
model.add(keras.layers.Dense(128, activation="sigmoid"))
model.add(keras.layers.Dense(128, activation="sigmoid"))
model.add(keras.layers.Dense(10, activation="sigmoid"))


# In[73]:


optimizer = keras.optimizers.SGD(lr=0.1)
model.compile(optimizer, loss="categorical_crossentropy", metrics=["accuracy"])


# In[74]:


model.summary()


# In[75]:


model.fit(x_train, y_train, batch_size=32, epochs=15)


# In[71]:


model.evaluate(x_test, y_test)

