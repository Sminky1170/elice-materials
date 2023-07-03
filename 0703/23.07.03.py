#######################3 [실습1] MNIST 분류모델 만들기
import numpy as np
import tensorflow as tf

# 동일한 실행 결과 확인을 위한 코드입니다.
np.random.seed(123)
tf.random.set_seed(123)

def load_data():
    # MNIST 데이터 세트를 불러옵니다.
    mnist = tf.keras.datasets.mnist

    # MNIST 데이터 세트를 Train set과 Test set으로 나누어 줍니다.

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # 지시사항1: 데이터를 지시사항에 따라 train data와 test data로 분리하세요
    train_images, train_labels = train_images[:5000], train_labels[:5000]
    test_images, test_labels = test_images[:1000], test_labels[:1000]


    return train_images, train_labels, test_images, test_labels

def preprocess_data(images, labels):
    # images와 labels를 전처리하여 반환합니다.
    ret_images = images / 255.0
    ret_images = tf.expand_dims(ret_images, -1)
    ret_labels = tf.one_hot(labels, depth=10)

    return ret_images, ret_labels

def get_model():
    # 지시사항2: 지시사항을 보고 조건에 맞는 모델을 정의하여 반환압니다.
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation="relu"))
    model.add(tf.keras.layers.Dense(32, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model

def main():
    # 데이터 불러오기
    train_images, train_labels, test_images, test_labels = load_data()

    # 데이터 전처리
    train_images, train_labels = preprocess_data(train_images, train_labels)

    test_images, test_labels = preprocess_data(test_images, test_labels)
    
    # 지시사항3: get_model 함수에서 정의된 모델을 가져옵니다.
    model = get_model()
    model.compile(
        loss="categorical_crossentropy", 
        optimizer="adam", 
        metrics=["accuracy"]
    )
    
    # 모델 학습을 시작합니다.
    history = model.fit(
        train_images,
        train_labels,
        epochs=20,
        batch_size=512,
        validation_data=(test_images, test_labels)
    )
    
    # Test 테이터로 모델을 평가합니다.
    loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)



    print("\nTest Loss : {:.4f} | Test Accuracy : {:.4f}%".format(loss, test_acc*100))
    
    # 모델의 학습 결과를 반환합니다.
    return history

######################## [실습2] CNN 모델 학습과 응용
import numpy as np
import tensorflow as tf
from utils import preprocess, visualize
# 동일한 실행 결과 확인을 위한 코드입니다.
np.random.seed(123)
tf.random.set_seed(123)


# TODO: 지시사항의 구조를 보고 CNN함수를 완성하세요
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


# TODO: 전달받은 결과를 보고 해당 숫자가 홀수이면 True, 짝수이면 False를 return 합니다.
def is_odd(model, image):
    image = tf.expand_dims(image, 0)
    pred = model.predict(image)
    result = np.argmax(pred[0])
    if result % 2 == 1:
        return True
    else:
        return False

def main():
    # MNIST dataset을 전처리한 결과를 받아옵니다.
    train_images, test_images, train_labels, test_labels = preprocess()

    # CNN()에서 정의한 모델을 불러와 model에 저장합니다.
    model = CNN()

    # TODO: 지시사항을 보고 model을 compile합니다.
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # TODO: 학습 결과를 history라는 변수에 저장합니다.
    history = model.fit(
        train_images,
        train_labels,
        epochs = 2,
        validation_data=(test_images, test_labels),

    )
    
    # Test 테이터로 모델을 평가합니다.
    loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    print("\nTest Loss : {:.4f} | Test Accuracy : {:.4f}%".format(loss, test_acc*100))


    # 첫번째 test_images를 시각화합니다.

    visualize(test_images[0])
    
    # 학습된 모델을 이용하여 홀수인지 판단하는 과정을 구현합니다.
    # is_odd()를 구현하지 못했다면 일단 main함수가 실행되기 위해 삭제하셔도 됩니다.
    odd = is_odd(model, test_images[0])
    print(f"입력한 숫자는 {'홀수'if odd else '짝수'}입니다.")
    
    # 학습 결과 history를 반환합니다.
    return history
        
if __name__ == "__main__":
    main()
######################## [실습3] 학습을 위한 이미지 데이터 증강
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from elice_utils import EliceUtils

elice_utils = EliceUtils()

# example of horizontal shift image augmentation
from numpy import expand_dims
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

def data_augmenter(mode=0):
    if mode == 0:
        # ToDo : 너비를 기준으로 shfit하는 augmentation을 설정합니다.

        datagen = ImageDataGenerator(width_shift_range=[-200,200])
    
    elif mode == 1:
        # ToDo: 회전하는 augmentation을 설정합니다.
        datagen = ImageDataGenerator(rotation_range=90)
    
    else:
        # ToDo: 밝기를 변화시키는 augmentation을 설정합니다.
        datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
        
        
    if datagen is not None:
        return datagen
        
    else:
        print('Daga Augmentation이 설정되지 않았습니다.')



def visualizer(img, datagen):
    # 이미지를 불러옵니다.
    data = img_to_array(img)    
    samples = expand_dims(data, 0)

    it = datagen.flow(samples, batch_size=1)
    
    # 이미지를 augmentation 결과에 따라 시각화합니다.
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        batch = it.next()
        image = batch[0].astype('uint8')
        pyplot.imshow(image)
    
    pyplot.savefig('result.png')
    elice_utils.send_image('result.png')



def main():

    img = load_img('kitty.png')
    
    # mode 0, 1, 2를 바꾸어 augmentation의 동작을 달리 해보세요.
    datagen = data_augmenter(mode=0) 
    
    # 코드가 작동한다고 판단되면 아래 주석을 해제해 결과를 확인해 보세요.
    visualizer(img, datagen)


if __name__ == "__main__":
    main()

######################## [실습4] Transfer Learning(1) 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from elice_utils import EliceUtils
elice_utils = EliceUtils()

# 시각화 함수
def Visulaize(histories, key='loss'):
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.savefig("plot.png")
    elice_utils.send_image("plot.png")


def main():

    # MNIST 데이터 세트를 불러오고 Train과 Test를 나누어줍니다.
    mnist = np.load('./data/mnist.npz')
    X_train, X_test, y_train, y_test = mnist['x_train'][:5000], mnist['x_test'][:1000], mnist['y_train'][:5000], mnist['y_test'][:1000]

    # Transfer Learning을 위해 MNIST 데이터를 나누어줍니다.
    # Label값 (0 ~ 4 / 5 ~ 9)에 따라 5개씩 나누어줍니다.
    x_mnist_04 = []
    y_mnist_04 = []
    x_mnist_59 = []
    y_mnist_59 = []

    for idx, label in enumerate(y_train):
        if label <= 4:
            x_mnist_04.append(X_train[idx])
            y_mnist_04.append(y_train[idx])

        else:
            x_mnist_59.append(X_train[idx])
            y_mnist_59.append(y_train[idx])

    # (0 ~ 4)의 데이터로 학습하고 (5 ~ 9)의 데이터로 검증을 해보겠습니다.
    X_train04, y_train04 = np.array(x_mnist_04), np.array(y_mnist_04)
    X_test59, y_test59 = np.array(x_mnist_59), np.array(y_mnist_59)

    # 나눈 MNIST 데이터 전처리
    X_train04 = X_train04.astype(np.float32) / 255.
    X_test59 = X_test59.astype(np.float32) / 255.

    X_train04 = np.expand_dims(X_train04, axis=-1)
    X_test59 = np.expand_dims(X_test59, axis=-1)

    y_train04 = to_categorical(y_train04, 10)
    y_test59 = to_categorical(y_test59, 10)

    # CNN 모델 선언
    CNN_model = keras.Sequential([
        keras.layers.Conv2D(32 ,kernel_size = (3,3), strides = (2,2), padding = 'same', activation=tf.nn.relu, input_shape=(28,28,1)),
        keras.layers.Conv2D(64 ,kernel_size = (3,3), strides = (2,2), padding = 'same', activation=tf.nn.relu),
        keras.layers.Conv2D(64 ,kernel_size = (3,3), strides = (2,2), padding = 'same', activation=tf.nn.relu),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation=tf.nn.sigmoid),
        keras.layers.Dense(16, activation=tf.nn.sigmoid),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # CNN model을 학습시켜줍니다.
    CNN_model.compile(optimizer='adam',loss='categorical_crossentropy', metrics = ['accuracy'])
    CNN_model.summary()

    # TODO : [0 ~ 4] Label의 데이터로 `CNN_model`을 학습시키고 [5 ~ 9] Label의 데이터로 `CNN_model`을 검증해보세요.
    CNN_history = CNN_model.fit(X_train04, y_train04,epochs= 20, batch_size = 100, validation_data=(X_test59, y_test59), verbose=2)

    # 각 모델 별 Loss 그래프를 그려줍니다.
    Visulaize([('CNN', CNN_history)])


    ################################################################################
    # Transfer Learning을 위한 과정입니다.
    # 학습된 CNN_model의 Classifier 부분인 Flatten() - Dense() layer를 제거해줍니다.
    CNN_model.summary()
    # TODO : Classifier 부분을 지워주세요.
    # 총 3개의 Dense layer와 1개의 Flatten layer가 있으므로 4번 pop을 해줍니다.
    for i in range(4):
        CNN_model.pop()

    # Classifier를 지운 모델의 구조를 확인합니다.
    CNN_model.summary()

    # 이제 CNN_model에는 학습된 Convolution Layer만 남아있습니다.

    # TODO : Convolution Layer의 학습된 Weight들을 저장합니다.
    CNN_model.save_weights('CNN_model.h5', save_format='h5')
    # 여기까지가 Transfer Learning의 1차 과정입니다.
    # 다음 실습에서 이어서 Transfer Learning을 진행하겠습니다.

    return CNN_model.summary()

if __name__ == "__main__":
    main()
######################## [실습5] Transfer Learning(2)
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from elice_utils import EliceUtils
elice_utils = EliceUtils()

# 시각화 함수
def Visulaize(histories, key='loss'):
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.savefig("plot.png")
    elice_utils.send_image("plot.png")


def main():

    # MNIST Data를 Train과 Test로 나누어줍니다.
    mnist = np.load('./data/mnist.npz')
    X_train, X_test, y_train, y_test = mnist['x_train'][:500], mnist['x_test'][:500], mnist['y_train'][:500], mnist['y_test'][:500]

    # MNIST Data를 전저리합니다.
    X_train = X_train.astype(np.float32) / 255.
    X_test = X_test.astype(np.float32) / 255.

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # 이전 실습에서 사용했던 CNN_model과 같은 구조를 가진 모델을 선언합니다.
    # 저장된 Weights를 불러오기 위해서는 모델의 구조가 같아야합니다.
    Transfer_model = keras.Sequential([
        keras.layers.Conv2D(32 ,kernel_size = (3,3), strides = (2,2), padding = 'same', activation=tf.nn.relu, input_shape=(28,28,1)),
        keras.layers.Conv2D(64 ,kernel_size = (3,3), strides = (2,2), padding = 'same', activation=tf.nn.relu),
        keras.layers.Conv2D(64 ,kernel_size = (3,3), strides = (2,2), padding = 'same', activation=tf.nn.relu)
    ])

    # TODO : Transfer_model 모델에 학습된 Weight를 넣어주세요.
    Transfer_model.load_weights('./data/CNN_model.h5')

    # TODO : 새로운 Classifier를 Transfer_model에 붙여주세요.
    Transfer_model.add(keras.layers.Flatten())
    Transfer_model.add(keras.layers.Dense(128, activation=tf.nn.sigmoid))
    Transfer_model.add(keras.layers.Dense(64, activation=tf.nn.sigmoid))
    Transfer_model.add(keras.layers.Dense(32, activation=tf.nn.sigmoid))
    Transfer_model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

    # Transfer_model을 출력합니다.
    Transfer_model.summary()


    # 전체 모델에서 Classifier 부분만 학습하기 위해 Trainable 여부를 설정할 수 있습니다.
    # TODO : 앞의 Convolution layer는 학습에서 제외하고 뒤의 Classifier 부분만 학습하기 위해 Trainable을 알맞게 설정해주세요.
    for layer in Transfer_model.layers[:3]:
        layer.trainable=False
    for layer in Transfer_model.layers[3:]:
        layer.trainable=True

    # Transfer_model을 학습시켜줍니다.
    Transfer_model.compile(optimizer='adam',loss='categorical_crossentropy', metrics = ['accuracy'])
    Transfer_history = Transfer_model.fit(X_train, y_train, epochs= 20, batch_size = 100, validation_data=(X_test, y_test), verbose=2)

    Visulaize([('CNN', Transfer_history)])

    # evaluate 함수를 사용하여 테스트 데이터의 결과값을 저장합니다.
    loss, test_acc = Transfer_model.evaluate(X_test, y_test, verbose = 0)


    return test_acc
    
if __name__ == "__main__":
    main()

