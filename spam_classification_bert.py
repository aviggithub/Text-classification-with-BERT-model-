# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 00:21:26 2021
pip install tensorflow-hub
pip install tensorflow-text

Tensorflow_hub: Place where all TensorFlow pre-trained models are stored.
Tensorflow: For model creation
Pandas: For data loading, manipulation and wrangling.
Tensorflow_text: Allows additional NLP text processing capabilities outside the scope of tensorflow.
Skelarn: For doing data splitting
Matplotlib: For visualization support
need to use google gpu

@author: avinash
"""

import tensorflow_hub as hub

import pandas as pd

import tensorflow_text as text

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

import numpy as np

# load data
df = pd.read_csv('spam_data_set.csv')
df.head()
#Spam Detection Exploratory Data Analysis
# check count and unique and top values and their frequency
df['Category'].value_counts()

#Clearly, the data is imbalanced and there are more good emails(ham) than spam emails. This may lead to a problem as a model may learn all the features of the ham emails over spam emails and thus always predict all emails as ham(OVERFITTIN!).

#Downsampling Data

# check percentage of data - states how much data needs to be balanced
print(str(round(747/4825,2))+'%')

# creating 2 new dataframe as df_ham , df_spam

df_spam = df[df['Category']=='spam']

df_ham = df[df['Category']=='ham']

print("Ham Dataset Shape:", df_ham.shape)

print("Spam Dataset Shape:", df_spam.shape)

# downsampling ham dataset - take only random 747 example
# will use df_spam.shape[0] - 747
df_ham_downsampled = df_ham.sample(df_spam.shape[0])
df_ham_downsampled.shape

# concating both dataset - df_spam and df_ham_balanced to create df_balanced dataset
df_balanced = pd.concat([df_spam , df_ham_downsampled])

df_balanced['Category'].value_counts()

df_balanced.sample(5)

# creating numerical repersentation of category - one hot encoding
df_balanced['spam'] = df_balanced['Category'].apply(lambda x:1 if x=='spam' else 0)

# displaying data - spam -1 , ham-0
df_balanced.sample(4)

#Train Test Split Strategy:
# loading train test split
from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(df_balanced['Message'], df_balanced['spam'],
                                                    stratify = df_balanced['spam'])

# downloading preprocessing files and model
bert_preprocessor = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')


text_input = tf.keras.layers.Input(shape = (), dtype = tf.string, name = 'Inputs')
preprocessed_text = bert_preprocessor(text_input)
embeed = bert_encoder(preprocessed_text)
dropout = tf.keras.layers.Dropout(0.1, name = 'Dropout')(embeed['pooled_output'])
outputs = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'Dense')(dropout)
# creating final model
model = tf.keras.Model(inputs = [text_input], outputs = [outputs])

# check the summary of the model
model.summary()

#model building, we will now compile our model using adam as our optimizer and binary_crossentropy as our loss function. /For metrics, we will use accuracy, precession, recall, and loss
Metrics = [tf.keras.metrics.BinaryAccuracy(name = 'accuracy'),
           tf.keras.metrics.Precision(name = 'precision'),
           tf.keras.metrics.Recall(name = 'recall')
           ]

# compiling our model
model.compile(optimizer ='adam',
               loss = 'binary_crossentropy',
               metrics = Metrics)

history = model.fit(X_train, y_train, epochs = 10)

# Evaluating performance
model.evaluate(X_test,y_test)

# getting y_pred by predicting over X_text and flattening it
y_pred = model.predict(X_test)
y_pred = y_pred.flatten() # require to be in one-dimensional array , for easy manipulation

# importing confusion maxtrix

from sklearn.metrics import confusion_matrix , classification_report

# creating confusion matrix 

cm = confusion_matrix(y_test,y_pred)

# plotting as a graph - importing seaborn
import seaborn as sns
# creating a graph out of confusion matrix
sns.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# printing classification report
print(classification_report(y_test , y_pred))

text_input = [" hello play rummy and win 5000 daily"]
 
predicted_results = model.predict(text_input)
output_class = np.where(predicted_results>0.5,'spam', 'ham')
print(output_class)