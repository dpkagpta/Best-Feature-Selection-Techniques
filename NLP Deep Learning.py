#!/usr/bin/env python
# coding: utf-8

# In[155]:


# importing the dependencies:

import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Embedding, Conv1D, GlobalMaxPool1D, SpatialDropout1D
from keras.callbacks import ModelCheckpoint
import os
import sklearn.metrics
from sklearn.metrics import roc_auc_score


# In[156]:


# variables:

output_dir = 'imdb/deep_net'
epochs = 4
batch_size = 128
n_dim = 64
n_unique_words = 5000
n_words_to_skip = 50
max_review_length = 100
pad_type = trunc_type = 'pre'
n_dense= 64
dropout=0.5
drop_embed = 0.2
n_conv = 256
k_conv = 3


# In[157]:


(x_train, y_train), (x_valid, y_valid) = imdb.load_data(num_words=n_unique_words, skip_top=n_words_to_skip)


# In[128]:


for x in x_train[0:6]:
    print(len(x))


# In[129]:


word_index=keras.datasets.imdb.get_word_index()


# In[130]:


for n,k in word_index.items():
    if k ==2:
        print(n)


# In[131]:


word_index = {k:v+3 for k,v in word_index.items()}


# In[132]:


word_index['PAD'] = 0
word_index['START'] = 1
word_index['UNK'] = 2


# In[ ]:





# In[133]:


def index_word(k):
    for n, o in word_index.items():
        if o==k:
            return n


# In[134]:


review = ' '.join(index_word(k) for k in x_train[0])
print(review)


# In[135]:


(all_X_train, _), (all_X_valid, _) = imdb.load_data()
full_review = ' '.join(index_word(l) for l in all_X_train[0])
print(full_review)


# In[136]:


x_train = pad_sequences(x_train, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0 )


# In[137]:


x_valid = pad_sequences(x_valid, maxlen=max_review_length, padding=pad_type, truncating=trunc_type, value=0 )


# In[138]:


for x in x_train[0:6]:
    print(len(x))


# In[139]:


review = ' '.join(index_word(k) for k in x_train[0])
review


# In[140]:


# Using Deep Neural Network


# In[141]:


model = Sequential()
model.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
model.add(Flatten())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))


# In[142]:


model.summary()


# In[143]:


modelcheckpoint = ModelCheckpoint(filepath=output_dir + '\weights{epoch:02d}.hdf5')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    


# In[144]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[145]:


model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.2, 
         callbacks=[modelcheckpoint])


# In[146]:


y_hat = model.predict_proba(x_valid)


# In[151]:


pct_auc = roc_auc_score(y_valid, y_hat) * 100
print('{:0.2f}'.format(pct_auc))


# In[ ]:





# In[147]:


# using convolutional neural network


# In[148]:


modelc=Sequential()
modelc.add(Embedding(n_unique_words, n_dim, input_length=max_review_length))
modelc.add(SpatialDropout1D(drop_embed))
modelc.add(Conv1D(n_conv, k_conv, activation='relu'))
modelc.add(GlobalMaxPool1D())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))


# In[149]:


model.compile(loss='binary_crossentropy', optimizer='adam',
    metrics=['accuracy'])


# In[150]:


model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
    verbose=1, validation_split=.20,
    callbacks=[modelcheckpoint])


# In[154]:


yc_hat = model.predict_proba(x_valid)
pct_auc = roc_auc_score(y_valid, yc_hat) * 100
print('{:0.2f}'.format(pct_auc))


# In[ ]:





# In[ ]:




