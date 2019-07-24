import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# read training data 
train = pd.read_csv('input/train.csv')

# read testing data
test= pd.read_csv('input/test.csv')

# put labels into y_train variable
Y_train = train["label"]
# Drop 'label' column
X_train = train.drop(labels = ["label"], axis = 1)

# Normalize the data
X_train = X_train / 255.0
test = test / 255.0

# Reshape
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
print("X_train shape: ",X_train.shape)
print("test shape: ",test.shape)

Y_train = to_categorical(Y_train, num_classes = 10)


# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1)

model = Sequential()

model.add(Conv2D(32, (5,5), activation ='relu', input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), activation ='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), activation ='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(32, activation = "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(10, activation = "softmax"))

# Compile the model
model.compile(optimizer = 'adam' , 
              loss = "categorical_crossentropy", 
              metrics=["accuracy"])

# data augmentation
datagen = ImageDataGenerator(
        rotation_range=10, # randomly rotate images in the range 5 degrees
        zoom_range = 0.1, # Randomly zoom image 5%
        width_shift_range=0.1, # randomly shift images horizontally 5%
        height_shift_range=0.1) # randomly shift images vertically 5%

datagen.fit(X_train)

callbacks = [EarlyStopping(monitor='acc', patience=15),
             ReduceLROnPlateau(patience=15, verbose=1),
             ModelCheckpoint('best_model.h5', monitor='val_acc',
                             verbose=2, save_best_only=True)]

# Fit the model
history = model.fit_generator(
        datagen.flow(X_train,Y_train, batch_size=200),
        epochs = 45, 
        validation_data = (X_val,Y_val),
        callbacks=callbacks,
        steps_per_epoch=X_train.shape[0] // 200)

history_df = pd.DataFrame(history.history)
history_df[['acc', 'val_acc']].plot()

# load beat weights
model.load_weights('best_model.h5')

result=np.argmax(model.predict(test), axis=1)
result = pd.Series(result, name="Label")
submit = pd.concat([pd.Series(range(1,28001),name = "ImageId"),result],
                    axis = 1)
submit.to_csv('submission.csv', index=False)
