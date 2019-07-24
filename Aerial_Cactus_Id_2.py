import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau


train_id = pd.read_csv('input/train.csv')
train_id['has_cactus'] = train_id['has_cactus'].astype(str)
test_id = pd.read_csv('input/sample_submission.csv')
train_dir = 'input/train/train'
test_dir = 'input/test/test'

# Construct model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPool2D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer = 'adam' , 
              loss = "binary_crossentropy", 
              metrics=["accuracy"])


# Data augmentation
datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.10,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=10, # randomly rotate images in the range 5 degrees
    zoom_range = 0.1) # Randomly zoom image 5%


train_gen = datagen.flow_from_dataframe(
    dataframe=train_id, 
    directory=train_dir, 
    x_col="id", 
    y_col="has_cactus",
    target_size=(32,32),
    subset="training",
    batch_size=200,
    class_mode="binary")  

validation_gen = datagen.flow_from_dataframe(
    dataframe=train_id,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(32,32),
    subset="validation",
    batch_size=200,
    class_mode="binary")

testgen = ImageDataGenerator(rescale=1/255)

test_gen = testgen.flow_from_dataframe(
        dataframe=test_id,
        directory=test_dir,
        x_col="id",
        target_size=(32,32),
        batch_size=1,
        shuffle=False,
        class_mode=None)

# =============================================================================
class myLogger(Callback):
    best_sum = 0
    def on_epoch_end(self, epoch,logs):
        this_sum = logs['acc'] + logs['val_acc']
        print('Sum of acc and val_acc =', round(this_sum, 5), '.')
    
        if this_sum > self.best_sum:
            print('Improved from ', round(self.best_sum, 5),
                  ' to ', round(this_sum, 5), '.')
            self.best_sum = this_sum
            model.save('best_model.h5')
        else:
            print('Did not improve from ', round(self.best_sum, 5), '.')

callbacks = [EarlyStopping(monitor='acc', patience=15),
             ReduceLROnPlateau(patience=15, verbose=1),
             myLogger()]
# =============================================================================

# fit the model
history = model.fit_generator(
        train_gen,
        steps_per_epoch=300,
        epochs=40,
        validation_data=validation_gen,
        validation_steps=validation_gen.n//validation_gen.batch_size,
        verbose=2,
        callbacks=callbacks)

history_df = pd.DataFrame(history.history)
history_df[['acc', 'val_acc']].plot()

# load beat weights
model.load_weights('best_model.h5')

# predict the result
result = model.predict_generator(
        test_gen,
        steps=test_gen.n)
test_id['has_cactus'] = result
test_id.to_csv('submission.csv', index = False)
