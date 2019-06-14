from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from parallel_gpu_efficient import *
import time

class cifar100vgg:
    def __init__(self,train=True):
        self.num_classes = 100
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar100vgg.h5')
            print("load model")

    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(name='bn_1'))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(name='bn_2'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(name='bn_3'))
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(name='bn_4'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(name='bn_5'))
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(name='bn_6'))
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(name='bn_7'))

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        print(mean)
        print(std)
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 121.936
        std = 68.389
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20

        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)


        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)


        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)



        #optimization details
        sgd = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)
        model.save_weights('cifar100vgg.h5')
        return model

    def get_length_model(self):
        return len(self.model.layers)

    def submodel(self,number_layer=3):
        # self.model.summary()
        out_flatten = Flatten()(self.model.layers[number_layer].output)
        new_model = Model(inputs=[self.model.input],
                          output=out_flatten)  # assuming you want the 3rd layer from the last
        # out = keras.layers.MaxPool2D((3, 3))(base_model.output)
        # out = Dense(fc1_size, activation='sigmoid')(out)
        return new_model


def normalize_production(x):
    # this function is used to normalize instances in production according to saved training set statistics
    # Input: X - a training set
    # Output X - a normalized training set according to normalization constants.

    # these values produced during first training and are general for the standard cifar10 training set normalization
    mean = 120.707
    std = 64.15
    return (x - mean) / (std + 1e-7)

if __name__ == '__main__':
    model = cifar100vgg(False)
    m = model.build_model()
    m.summary()
    print(model.get_length_model())
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = normalize_production(x_train)
    x_test = normalize_production(x_test)
    #x_train,x_test =    model.normalize(x_train,x_test)
    y_train = keras.utils.to_categorical(y_train, 100)
    y_test = keras.utils.to_categorical(y_test, 100)
    y_test_one_hot = np.argmax(y_test,1)
    y_train_one_hot = np.argmax(y_train,1)
    x_train_flatten = x_train.reshape(x_train.shape[0], -1)
    x_test_flatten = x_test.reshape(x_test.shape[0], -1)

    np.random.seed(12)
    [number, size] = x_train_flatten.shape
    x_random = np.random.rand(number,size)

    # for i in range(5):
    #     name = 'result_' + str(i) + '.txt'
    # stability(x_train_flatten,y_train_one_hot)

    # NUM = 2_000
    # x_train = x_train[:NUM, ]
    # x_test = x_test[:NUM, ]
    # y_train = y_train[:NUM, ]
    # y_test = y_test[:NUM, ]
    # y_test_one_hot = y_test_one_hot[:NUM]
    # y_train_one_hot = y_train_one_hot[:NUM]

    print("predict(train)")
    predicted_x = model.predict(x_train, normalize=False)
    residuals = np.argmax(predicted_x, 1) != np.argmax(y_train, 1)
    loss = sum(residuals)/len(residuals)
    print("loss (train): ",loss)
    #
    equal = np.argmax(predicted_x, 1) == np.argmax(y_train, 1)
    acc = sum(equal) / len(equal)
    print('accuracy (train): ', acc)

    print("predict(test)")
    predicted_x = model.predict(x_test, normalize=False)
    residuals = np.argmax(predicted_x, 1) != np.argmax(y_test, 1)
    loss = sum(residuals)/len(residuals)
    print("loss (test): ",loss)
    #
    equal = np.argmax(predicted_x, 1) == np.argmax(y_test, 1)
    acc = sum(equal) / len(equal)
    print('accuracy (test): ', acc)

    output_file_train = open('cifar100_train.txt', 'a+', 1)
    output_file_test = open('cifar100_test.txt', 'a+', 1)
    model.summary()
    for i in range(40, 100):
        # tf.reset_default_graph()
        print("_____________________________________________________")
        model = cifar100vgg(False)
        if i >= 53:
            sub_model = model.submodel(i, False)
        else:
            sub_model = model.submodel(i)
        #
        x_train_out = sub_model.predict(x_train)
        x_test_out = sub_model.predict(x_test)
        print('predict: ', i, x_train_out.shape, x_test_out.shape)

        # print('calculate si test DR')
        # prev = time.time()
        # result2, number2 = DR_SepI(x_test_out, y_test_one_hot)
        # output_file_test_DR.write("%i %f %i %f\n" % (i, result2, number2, float(result2/number2)))
        # new = time.time()
        # print(i,result2,number2,float(result2/number2),new-prev)
        # print('calculate SI train DR')
        # result1, number1 = DR_SepI(x_train_out, y_train_one_hot)
        # output_file_train_DR.write("%i %f %i %f\n" % (i, result1, number1, float(result1/number1)))
        # print(i,result1,number1,float(result1/number1),time.time()-prev)

        print('calculate SI test')
        # prev = time.time()
        result2, number2 = calculate_SI(x_test_out, y_test_one_hot)
        output_file_test.write("%i %f %i %f\n" % (i, result2, number2, float(result2 / number2)))
        # new = time.time()
        print(i, result2, number2, float(result2 / number2))
        # prev = new
        # K.clear_session() #after call calculate_SI must clear session
        prev = 0
        print('calculate SI train')
        result1, number1 = calculate_SI(x_train_out, y_train_one_hot)
        output_file_train.write("%i %f %i %f\n" % (i, result1, number1, float(result1 / number1)))
        print(i, result1, number1, float(result1 / number1), time.time() - prev)
        x_train_out = None
        x_test_out = None
        sub_model = None
        K.clear_session()  # after call calculate_SI must clear session


