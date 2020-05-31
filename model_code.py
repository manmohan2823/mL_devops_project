#Loading the dataset. Here we are going to use the MNIST dataset to create a model. 

import keras
from keras.datasets import mnist
dataset=mnist.load_data()


#Splitting the dataset into appropriate variables

train, test = dataset
X_train, y_train = train
X_test, y_test = test

#Let us reshape the images

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Performing one-hot encoding to convert categorical variables into dummy

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.optimizers import RMSprop


model = Sequential()
model.add(Convolution2D(filters = 2, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
parameter=1
filt=6
for parameter in range(parameter):
    model.add(Convolution2D(filters=filt, kernel_size=(3,3), activation='relu'))
    filt=filt*2
model.add(Flatten())
model.add(Dense(units=10, activation='softmax')

#compiling the model

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accurate'])

#fiting the model
model.fit(X_train, y_train, epochs=1,verbose=1)

#checking the accuracy

a=model.history.history.get('accurate')
a= a[0] * 100
a=int(a)


print("Accurcay of the model is:", a)

#storing the accuracy

accurate= open('/root/accurate.txt','w+')
accurate.write(str(a))
accurate.close()

display_accuracy = open('/home/show.html','r+')
display_accuracy.read()

# display_accuracy
display_accuracy.write('\nAccuracy achieved : ' + str(a) +'\n</pre>')
display_accuracy.close()
    
