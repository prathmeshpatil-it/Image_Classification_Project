
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

model=Sequential([
Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
MaxPooling2D(),
Flatten(),
Dense(128,activation='relu'),
Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=1)
model.save('models/image_classifier.h5')
