
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop 



from keras.utils import np_utils
import keras
import numpy as np

classes = ["石原さとみ", "新垣結衣", "北川景子", "小松菜奈"]
num_classes = len(classes)
image_size = 64


def load_data():
    X_train, X_test, y_train, y_test = np.load("./女優.npy", allow_pickle=True)
    X_train = X_train.astype("float") / 255
    X_test  = X_test.astype("float") / 255
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test  = np_utils.to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test


def train(X, y, X_test, y_test):
    model = Sequential()


    model.add(Conv2D(32,(3,3), padding='same',input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))
    model.add(Dense(4)) 
    model.add(Activation('softmax'))

    opt = RMSprop(lr=0.00005, decay=1e-6)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.fit(X, y, batch_size=28, epochs=40)
    model.save('./cnn.h5')

    return model

def main():
    X_train, y_train, X_test, y_test = load_data()

    model = train(X_train, y_train, X_test, y_test)

main()
