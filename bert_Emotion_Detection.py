#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install transformers')
''' 
IMPORT THE APPROPRIATE LIBARYS
'''
import os
import pandas as pd
import gensim 
import numpy as np
import keras
import transformers
import tensorflow as tf
from transformers import BertModel, TFBertModel
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
from keras import layers
from keras import optimizers



embedding_size = 300
max_words = 5000
max_review_length = 70
prop_val = 0.2  
data = pd.read_csv('Downloads/tweet_emotions.csv/tweet_emotions.csv', on_bad_lines='skip')
import sys
print("Python version")
print (sys.version)
print("Version info.")
print (sys.version_info)


# In[ ]:


'''
the following code removes data 
'''
data = data[data["sentiment"] != "empty"];
data = data[data["sentiment"] != "neutral"];


# In[ ]:



print(data.sentiment.unique().tolist())


# In[15]:


totalRows = len(data.tweet_id)
print(totalRows)
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transformer_model = TFBertModel.from_pretrained("bert-base-uncased")


# 

# In[16]:


with tf.device("/device:GPU:0"):    


    import re
    sentiment = []
    tweet = []


    for i in range(totalRows):
        tweet_bert = data.iloc[i, 2]
        tweet_bert = remove_stopwords(tweet_bert)
        tweet_bert = re.sub('@[\w]+','',tweet_bert)
        tweet_bert = re.sub('=+','',tweet_bert)
        tweet_bert = re.sub('[^a-zA-Z0-9 \n\.]', '', tweet_bert)
        tweet_bert = str("[CLS] " + tweet_bert + "[SEP]" ) 
        tweet.append(tweet_bert)
        sentiment.append(data.iloc[i, 1])
    print(tweet[0])



  
      
        
        


    #for i in range(totalRows):
      #tokens = tokenizer.tokenize(tweet[i])  

np.random.seed(2)
np.random.shuffle(tweet)
np.random.seed(2)
np.random.shuffle(sentiment)




# In[17]:


from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
sentiment = encoder.fit_transform(sentiment)
print(sentiment)


# In[28]:


def getModel():

  from tensorflow.keras import regularizers

  input_ids = tf.keras.layers.Input(shape=(max_review_length,), name='input_token', dtype='int32')
  attention_mask = tf.keras.layers.Input(shape=(max_review_length,), name='attention_token', dtype='int32')
  token_type_ids = tf.keras.layers.Input(shape=(max_review_length,), name='token_type_ids_token', dtype='int32')

  x = transformer_model(input_ids, attention_mask)[0]
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dropout(0.8)(x)
  x = tf.keras.layers.Dense(64, bias_regularizer=regularizers.L2(1e-1))(x)
  x = tf.keras.layers.Dropout(0.8)(x)
  x = tf.keras.layers.Dense(11,activation='softmax')(x)


  model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs = x)   

  model.summary()
  model.compile(loss='CategoricalCrossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.00005), metrics=['acc'])
  my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=4)]
  return model


# In[29]:



x_train = tweet[100:]
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

with tf.device('/device:GPU:0'):
  len_val = int(len(tweet) * prop_val)
  x_train = tweet[len_val:]
  y_train = np.array(sentiment[len_val:])

  x_val = tweet[:len_val]
  y_val = np.array(sentiment[:len_val])

  x_train_tokenised = tokenizer(x_train, padding='max_length', max_length=max_review_length, truncation=True, return_tensors='tf')
  x_val_tokenised = tokenizer(x_val, padding='max_length', max_length=max_review_length, truncation=True, return_tensors='tf')

 
  x_train_inputs_ids = x_train_tokenised['input_ids']
  x_train_mask = x_train_tokenised['attention_mask']
  x_train_tokens = x_train_tokenised['token_type_ids']

  x_val_inputs_ids = x_val_tokenised['input_ids']
  x_val_mask = x_val_tokenised['attention_mask']
  x_val_tokens = x_val_tokenised['token_type_ids']
    

  network = getModel()

  hist = network.fit((x_train_inputs_ids, x_train_mask, x_train_tokens), 
      y_train, epochs=4, batch_size=64, 
      validation_data=((x_val_inputs_ids, x_val_mask, x_val_tokens),y_val))
        
  mae_hist = hist.history['acc']
  



# In[ ]:


network.save("bertModelReduced")
import sklearn
class_names = ['sadness', 'worry', 'surprise', 'love', 'fun', 'hate', 'happiness']
y_predict = network.predict((x_val_inputs_ids, x_val_mask, x_val_tokens))
print(encoder.inverse_transform(y_predict))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def process(text):
        import re
        
        from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
        text = remove_stopwords(text)
        text = re.sub('@[\w]+','',text)
        text = re.sub('=+','',text)
        text = re.sub('[^a-zA-Z0-9 \n\.]', '', text)
        return text;


# 

# In[ ]:



import transformers
import re
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = "i cant stand my hero academia. how do people watch this crap"
text = process(text)
text = str("[CLS] " + text + "[SEP]" ) 
text = tokenizer(text, padding='max_length', max_length=max_review_length, truncation=True, return_tensors='tf')
ids = text['input_ids']
print (ids.shape)
token = text['token_type_ids']
mask =text['attention_mask']

prediction = model.predict((ids, token, mask))
from sklearn import preprocessing

print(encoder.inverse_transform(prediction))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




