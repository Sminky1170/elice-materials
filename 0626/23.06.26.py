############################# [실습1] 데이터 전처리
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data.mnist import load_mnist
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import tensorflow as tf

from elice_utils import EliceUtils

elice_utils = EliceUtils()

def mnist_show():
    # TODO - Fashion MNIST 읽어오기

    (x_train, t_train), (x_test, t_test) = load_mnist()
    
    # TODO - 784를 (28, 28)로 변환
    x_train, x_test = x_train.reshape(-1, 28, 28), x_test.reshape(-1, 28, 28)
    
    # 클래스 이름
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # TODO - 사진 1장 확인
    pic_number = 0
    
    plt.figure()
    plt.imshow(x_train[pic_number], cmap='gray_r')
    plt.title(class_names[t_train[pic_number]])
    plt.colorbar()
    plt.grid(False)
    plt.savefig("result1.png")
    elice_utils.send_image("result1.png")
    plt.close()    
    
    
    # 인근 사진 확인
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i+pic_number], cmap='gray_r')
        plt.xlabel(class_names[t_train[i+pic_number]])
    plt.savefig("result1.png")
    elice_utils.send_image("result1.png")
    plt.close()
    
    return x_train, x_test


def main():

    # 이미지 확인
    verbose = 1         # 화면출력
    epochs = 5          # 반복횟수
    percentile = 2     # 훈련 데이터 세트의 크기 비율

    mnist_show()

    # MNIST 데이터 읽어들이기
    # reshape() 이용, 차원 맞추기
    (x_train, t_train), (x_test, t_test) = load_mnist()

    # 차원과 데이터 크기 조절
    num_train, num_test = percentile*600, percentile*100
    x_train, x_test = x_train.reshape(60000, 28, 28), x_test.reshape(10000, 28, 28)
    (x_train, t_train), (x_test, t_test) = (x_train[:num_train], t_train[:num_train]), (x_test[:num_test], t_test[:num_test])

    # 모델 만들기
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10,  activation='softmax')

    ])
    
    # optimizer = ['adam', 'sgd', 'adagrad']
    # loss = ['sparse_categorical_crossentropy', 'mse', 'mae']
    # metrics =['accuracy', 'MeanAbsoluteError', 'sparse_categorical_crossentropy']
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])


    # 모델 훈련
    history = model.fit(x_train, t_train, epochs=epochs, verbose=verbose)
    test_eval = model.evaluate(x_test,  t_test, verbose=verbose)


    result = history.history['accuracy'][-1]
    
    x = history.epoch
    y = history.history['loss']


    plt.plot(x, y, 'g+:')

    plt.savefig("result1.png")
    elice_utils.send_image("result1.png")
    plt.close()

    print('훈련자료에 대한 정확도 : {:.6f}'.format(result))
    print('검증자료에 대한 정확도 : {:.6f}, loss : {:.4f}'.format(test_eval[1], test_eval[0]))

    return result

if __name__ == "__main__":
    main()



######################################### [실습2] MLP vs CNN
import numpy as np
import tensorflow as tf
from Visualize import Visualize

# 동일한 실행 결과 확인을 위한 코드입니다.
np.random.seed(123)
tf.random.set_seed(123)

def preprocess():

    # MNIST 데이터 세트를 불러옵니다.
    mnist = tf.keras.datasets.mnist

    # MNIST 데이터 세트를 Train set과 Test set으로 나누어 줍니다.
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images, train_labels = train_images[:5000], train_labels[:5000]
    test_images, test_labels = test_images[:1000], test_labels[:1000]

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    train_images = tf.expand_dims(train_images, -1)
    test_images = tf.expand_dims(test_images, -1)

    train_labels = tf.one_hot(train_labels, depth=10)
    test_labels = tf.one_hot(test_labels, depth=10)

    return train_images, test_images, train_labels, test_labels


def MLP():
# 지시사항 1: 지시사항의 구조를 보고 MLP함수를 완성하세요
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    # 분류기 (classifier)
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model

# 지시사항 2: 지시사항의 구조를 보고 CNN함수를 완성하세요
def CNN():

    model = tf.keras.Sequential()
    
    # Feature Extractor
    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="SAME",
            input_shape=(28, 28, 1),
        )
    )
    model.add(tf.keras.layers.MaxPool2D(padding="SAME"))
    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="SAME"
        )
    )
    model.add(tf.keras.layers.MaxPool2D(padding="SAME"))
    model.add(
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="SAME"
        )
    )
    model.add(tf.keras.layers.MaxPool2D(padding="SAME"))
    model.add(tf.keras.layers.Flatten())
    
    # 분류기 (classifier)
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))


    return model

def main():

    train_images, test_images, train_labels, test_labels = preprocess()
    cnn_model = CNN()
    cnn_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    mlp_model = MLP()
    mlp_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]

    )
    
    # MLP모델부터 학습을 시작합니다.
    history_mlp = mlp_model.fit(
        train_images,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_images, test_labels),
    )
    
    # Test 테이터로 mlp 모델을 평가합니다.
    mlp_loss, mlp_test_acc = mlp_model.evaluate(test_images, test_labels, verbose=0)
    
    # CNN 모델 학습을 시작합니다.
    history_cnn = cnn_model.fit(
        train_images,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_images, test_labels),
    )
    
    # Test 테이터로 cnn 모델을 평가합니다.
    cnn_loss, cnn_test_acc = cnn_model.evaluate(test_images, test_labels, verbose=0)


    print("\nMLP Test Loss : {:.4f} | Test Accuracy : {:.4f}%".format(mlp_loss, mlp_test_acc*100))

    print("CNN Test Loss : {:.4f} | Test Accuracy : {:.4f}%".format(cnn_loss, cnn_test_acc*100))
    
    # 그래프로 CNN모델과 MLP 모델의 성능을 비교합니다.
    Visualize([("CNN", history_cnn), ("MLP", history_mlp)], "accuracy")


    return history_cnn
        
if __name__ == "__main__":
    main()


#############################################[실습3] CNN 모델 구성 실습
# main.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data.mnist import load_mnist
import img_show
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import tensorflow as tf

from elice_utils import EliceUtils

elice_utils = EliceUtils()


# 모델 만들기
# TODO : 클래스로 모델 만들기

class MyModel(Model):
    def __init__(self):

        super(MyModel, self).__init__()
        
        self.conv1 = Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.conv2 = Conv2D(64, 3, activation='relu')
        self.conv3 = Conv2D(64, 3, activation='relu')


        self.dens1 = Dense(64, activation='relu')

        self.dens2 = Dense(10, activation='softmax')
        
        self.flatten = Flatten()
        self.maxpooling = MaxPooling2D()
        
    def call(self, inputs, training=False):
    
        x = self.conv1(inputs)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = self.maxpooling(x)
        x = self.conv3(x)
        x = self.maxpooling(x)



        x = self.flatten(x)
        
        x = self.dens1(x)
        x = self.dens2(x)
        return x



def main():

    # TODO : 이미지 확인
    verbose = 1         # 화면출력
    epochs = 5          # 반복횟수
    percentile = 20     # 훈련 데이터 세트의 크기 비율

    img_show.mnist_show()


    # MNIST 데이터 읽어들이기
    # TODO : reshape() 이용, 차원 맞추기
    (x_train, t_train), (x_test, t_test) = load_mnist()

    # 차원과 데이터 크기 조절
    num_train, num_test = percentile*600, percentile*100
    x_train, x_test = x_train.reshape(60000, 28, 28, 1), x_test.reshape(10000, 28, 28, 1)
    (x_train, t_train), (x_test, t_test) = (x_train[:num_train], t_train[:num_train]), (x_test[:num_test], t_test[:num_test])



    # 모델 만들기

    model = MyModel()
    
    
    # optimizer = ['adam', 'sgd', 'adagrad']
    # loss = ['sparse_categorical_crossentropy', 'mse', 'mae']
    # metrics =['accuracy', 'MeanAbsoluteError', 'sparse_categorical_crossentropy']
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])



    # TODO : 모델 훈련
    history = model.fit(x_train, t_train, epochs=epochs, verbose=verbose)
    test_eval = model.evaluate(x_test,  t_test, verbose=verbose)

    elice_utils.send_file('data/input.txt')


    result = history.history['accuracy'][-1]
    
    x = history.epoch
    y = history.history['loss']


    plt.plot(x, y, 'g+:')

    plt.savefig("result1.png")
    elice_utils.send_image("result1.png")
    plt.close()

    print('훈련자료에 대한 정확도 : {:.6f}'.format(result))
    print('검증자료에 대한 정확도 : {:.6f}, loss : {:.4f}'.format(test_eval[1], test_eval[0]))

    return result

if __name__ == "__main__":
    main()


####################################################3[실습4] VGG NET
import tensorflow as tf
from tensorflow import keras
from elice_utils import EliceUtils
elice_utils = EliceUtils()

def VGG16():
    # Sequential 모델 선언
    model = keras.Sequential()
    # TODO : 3 x 3 convolution만을 사용하여 VGG16 Net을 완성해보세요.
    # 첫 번째 Conv Block
    # 입력 Shape는 ImageNet 데이터 세트의 크기와 같은 RGB 영상 (224 x 224 x 3)입니다.
    model.add(keras.layers.Conv2D(filters = 64, kernel_size = 3, activation=tf.nn.relu, padding='same', input_shape = (224, 224, 3)))
    model.add(keras.layers.Conv2D(filters = 64, kernel_size = 3, activation=tf.nn.relu, padding='same'))

    model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
    
    # 두 번째 Conv Block
    model.add(keras.layers.Conv2D(filters = 128, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 128, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
    
    # 세 번째 Conv Block
    model.add(keras.layers.Conv2D(filters = 256, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 256, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 256, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
    
    # 네 번째 Conv Block
    model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
    
    # 다섯 번째 Conv Block
    model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.Conv2D(filters = 512, kernel_size = 3, activation= tf.nn.relu, padding='same'))
    model.add(keras.layers.MaxPooling2D(pool_size = 2, strides = 2))
    
    # Fully Connected Layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation= tf.nn.relu))
    model.add(keras.layers.Dense(4096, activation= tf.nn.relu))
    model.add(keras.layers.Dense(1000, activation= tf.nn.softmax))
    
    return model


vgg16 = VGG16()
vgg16.summary()



######################################### [실습5] RESNET
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import add, Input,Dense,Activation, Flatten, Conv2D, MaxPooling2D, GlobalMaxPooling2D, ZeroPadding2D, AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model


# 입력과 출력의 Dimension이 같은 경우 사용합니다.

def identity_block(input_tensor, kernel_size, filters):
    
    filters1, filters2, filters3 = filters
    
    x = Conv2D(filters1, (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1))(x)

    x = BatchNormalization()(x)
    
    # 입력(x) : input_tensor와 F(x) : x를 더해줍니다.
    # TODO : add()와 Activation() 메서드를 사용해서 relu(F(x) + x) 의 형태로 만들어보세요.
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x



def residual_block(input_tensor, kernel_size, filters, strides=(2, 2)):

    filters1 , filters2 , filters3 = filters
    
    # 입력 Feature Map의 Size를 1/2로 줄이는 대신 Feature map의 Dimension을 2배로 늘려줍니다.
    x = Conv2D(filters1, (1, 1), strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1))(x)

    x = BatchNormalization()(x)
    
    # TODO : Projection Shortcut Connection을 구현해보세요.
    # 1 x 1 Convolution 연산을 수행하여 Dimension을 2배로 증가시키고
    # 입력 Feature map의 size를 1/2로 축소시켜보세요.
    shortcut = Conv2D(filters3, (1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)


    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50():
    # 입력 이미지의 Shape을 정해줍니다.
    shape = (224,224,3)

    inputs = Input(shape)
    
    # 입력 영상의 크기를 줄이기 위한 Conv & Max-pooling
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    # 첫 번째 Residual Block (입력 영상 Size 2배 축소 / Dimension 2배 증가)
    x = residual_block(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
    
    
    # 두 번째 Residual Block (입력 영상 Size 2배 축소 / Dimension 2배 증가)
    x = residual_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    
    # 세 번째 Residual Block (입력 영상 Size 2배 축소 / Dimension 2배 증가)
    x = residual_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    
    # 네 번째 Residual Block (입력 영상 Size 2배 축소 / Dimension 2배 증가)
    x = residual_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])


    # 마지막단에서 FC layer를 쓰지 않고 단순히 Averaging 합니다.
    x = AveragePooling2D((7, 7))(x)
    x = Flatten()(x)
    # 1000개의 Class 구분

    x = Dense(1000, activation='softmax')(x)
    
    # 모델 구성
    model = Model(inputs, x)
    return model


model = ResNet50()
model.summary()