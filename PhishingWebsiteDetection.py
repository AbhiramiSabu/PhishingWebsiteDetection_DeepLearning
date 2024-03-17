import numpy as np
import pandas as pd
df=pd.read_excel('Detection.xlsx')
df.head(10)
changes = {'Having @ Symbol': 'Have_Symbols',
        'Presence of IP Address': 'Have_IP',
        'Length of URL': 'URL_length',
        'No. of Slashes': 'No_of_slashes',
        'Special Character': 'Special_character',
        'No.of Dots': 'No_of_dots',
        'No. of Hyphen in Host Address': 'No_of_hyphen',
        '"Email" Keyword': 'Email_Keyword',
        'TLS ':'TLS',
        'Age of URL' : 'Age_of_URL',
        'Result': 'Label'
      }

# call rename () method
df.rename(columns=changes,
          inplace=True)
df
df['No_of_slashes'] = df['No_of_slashes'].replace([-1], 0)
df['Special_character'] = df['Special_character'].replace([-1], 0)
df['No_of_dots'] = df['No_of_dots'].replace([-1], 0)
df['No_of_hyphen'] = df['No_of_hyphen'].replace([-1], 0)
df['Email_Keyword'] = df['Email_Keyword'].replace([-1], 0)


df['Age_of_URL'] = df['Age_of_URL'].replace([-1], 0)
df['Label'] = df['Label'].replace([-1], 0)
df['TLS'] = df['TLS'].replace([-1], 0)
df
X=df.iloc[0: , 1:-1].values
y = df['Label']
X.shape
X
y.shape
df.info()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.20, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Embedding,SimpleRNN,LSTM
model = Sequential()
model.add(Embedding(input_dim=5000,output_dim=32,input_length=10))
model.add(SimpleRNN(units=32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
 metrics=['acc'])
model.summary()
batch_size = 32
model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, validation_split=0.2)
acc = model.evaluate(X_test,Y_test)
print("Test loss is {0:.2f} accuracy is {1:.2f} ".format(acc[0],acc[1]))
model.save_weights('basic_model.h5')

model.load_weights('basic_model.h5')
print("Accuracy of the model on Training Data is - " , model.evaluate(X_train,Y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,Y_test)[1]*100 , "%")
model_json=model.to_json()
with open("model.json","w") as json_file:
     json_file.write(model_json)

model.save_weights("RNN_model.h5")
print("Saved model to disk")
import tensorflow
from tensorflow.keras.models import model_from_json
json_file1=open('model.json','r')
loaded_model_json=json_file1.read()
json_file1.close()
loaded_model=model_from_json(loaded_model_json)

loaded_model.load_weights("RNN_model.h5")

print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
score=loaded_model.evaluate(X_test,Y_test,verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1],score[1]*100))
from keras.layers import GRU

model = Sequential()
model.add(Embedding(input_dim=5000,output_dim=32,input_length=10))

model.add(GRU(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, validation_split=0.2)
print("Accuracy of the model on Training Data is - " , model.evaluate(X_train,Y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,Y_test)[1]*100 , "%")
model_json=model.to_json()
with open("model.json","w") as json_file:
     json_file.write(model_json)

model.save_weights("GRU_model.h5")
print("Saved model to disk")
import tensorflow
from tensorflow.keras.models import model_from_json
json_file1=open('model.json','r')
loaded_model_json=json_file1.read()
json_file1.close()
loaded_model=model_from_json(loaded_model_json)

loaded_model.load_weights("GRU_model.h5")

print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
score=loaded_model.evaluate(X_test,Y_test,verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1],score[1]*100))
X_test[:10]
Y_test[:10]
ytest_data = np.array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1])
test_data = ytest_data.reshape((1, 10))
test_data
predictNextNumber = model.predict(test_data, verbose=1)
print(predictNextNumber)