from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.optimizers import SGD,Adam
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2


def get_model(weight_decay=0.0):
    model = Sequential()
    model.add(Conv2D(filters=20, kernel_size=[
              5, 5], activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(filters=50, kernel_size=[
              3, 3], activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    if weight_decay != 0.0 :
        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer'):
                layer.kernel_regularizer = l2(weight_decay)
    return model


def get_lrs(epoch):
    if epoch < 40:
        return float(0.02)
    elif epoch < 60:
        return float(0.005)
    else:
        return float(0.001)


def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train/255.
    x_test = x_test/255.
    weight_decay = 0.0   #5e-4
    model = get_model(weight_decay)
    sgd = SGD(lr=0.02, momentum=0.9,nesterov=True)
    # adam = Adam(lr=0.02)
    model.compile(
        optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lrs = LearningRateScheduler(get_lrs, verbose=1)
    model.fit(x=x_train, y=y_train, batch_size=128, epochs=80, verbose=1,
              shuffle=True, validation_data=(x_test, y_test), callbacks=[lrs])


if __name__ == '__main__':
    main()