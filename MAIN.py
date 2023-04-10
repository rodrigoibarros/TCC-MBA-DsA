# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:17:28 2023

@author: rodrigoibarros@usp.br
Rodrigo de Barros Iscuissati
"""
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
import keras
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Import database from .xlsx file
data = pd.read_excel('DataBase_TCC_MBA.xlsx')

# Data normalization
scaler = MinMaxScaler()
scaler.fit(data)
normalized = scaler.transform(data)
raw_data = normalized

# Split the databse in training and test data
TRAIN_TEST_SPLIT=0.8

mask = np.random.rand(len(raw_data)) < TRAIN_TEST_SPLIT
tr_dataset = raw_data[mask]
tr_data = tr_dataset[:,1:21]
tr_labels = tr_dataset[:,21:22]

te_dataset = raw_data[~mask]
te_data = te_dataset[:,1:21]
te_labels = te_dataset[:,21:22]

#Define te ANN model
H1=30 #1st layer
H2=10 #2nd layer
H3=5  #3rd layer

ffnn = Sequential()
ffnn.add(Dense(H1, input_shape=(20,),activation="relu"))
ffnn.add(Dense(H2, input_shape=(H1,),activation="relu"))
ffnn.add(Dense(H3, input_shape=(H2,),activation="relu"))
ffnn.add(Dense(1, activation="sigmoid"))

print('==================================================')
print('================ Model Summary ===================')
print('==================================================')
ffnn.summary()


#Training the ANN model
opt = keras.optimizers.Adam(learning_rate=0.01)
ffnn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'],)
metrics_fit = ffnn.fit(tr_data, tr_labels, epochs=150, batch_size=len(tr_data),verbose=0)

metrics_tr = ffnn.evaluate(tr_data, tr_labels, verbose=0)
print('==================================================')
print('=================== Training =====================')
print('==================================================')

print('Training Metrics')
print("%s: %.2f%%" % (ffnn.metrics_names[1],metrics_tr[1]*100))

#Test the ANN model
metrics = ffnn.evaluate(te_data, te_labels, verbose=0)
print('==================================================')
print('===================== Test =======================')
print('==================================================')

print('Test Metrics')
print("%s: %.2f%%" % (ffnn.metrics_names[1],metrics[1]*100))

#Making predictions with the model
predictions = ffnn.predict(te_data)
# round predictions 
rounded = [round(x[0]) for x in predictions]
res = []
for i in predictions:
    if i<0.5: 
        res.append(0)
    else:
        res.append(1)
        
#Create confusion matrix
result = confusion_matrix(te_labels, res)

print('==================================================')
print('================ Confusion Matrix ================')
print('==================================================')
print(result)
print('accuracy =', 100*(result[0][0]+result[1][1])/
      (result[0][0]+result[0][1]+result[1][0]+result[1][1]),'%')

print('recall(precision) = ',100*result[0][0]/
      (result[0][0]+result[1][0]),'%')

print('specificity = ',100*result[1][1]/
      (result[0][1]+result[1][1]),'%')

#Save ANNN model
ffnn.save("model_teste.h5")


