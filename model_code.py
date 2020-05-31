#Loading the dataset. Here we are going to use the MNIST dataset to create a model. 

# In[ ]:


#Loading the dataset. Here we are going to use the MNIST dataset to create a model. 
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
y_test = to_categorical(y_test)



from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import Flatten
model = Sequential()
model.add(Convolution2D(filters = 2, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
parameter=1
filt=6
for parameter in range(parameter):
    model.add(Convolution2D(filters=filt, kernel_size=(3,3), activation='relu'))
    filt=filt*2
model.add(Flatten())
model.add(Dense(units=10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#odelfitting

model.fit(X_train, y_train, validation_data=(x_test, y_test), epochs=1)


#storing accuracy in variable 

scores = model.evaluate(x_test, y_test)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


a=model.history.history.get('accuracy')
a= a[0] * 100
a=int(a)


#printing accuracy


print("Accurcay of the model is:", a)


#file handling


acc = open("accurate.txt", "w+")
acc.write(str(a))
acc.close()


display_accuracy = open('/home/show.html','r+')
display_accuracy.read()
display_accuracy.write('\nAccuracy achieved : ' + str(scores[1])+'\n</pre>')
display_accuracy.close()

