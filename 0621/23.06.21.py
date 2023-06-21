############################ [실습1] Softmax 함수
import numpy as np

# TODO : 비례 확률 함수 prop_function() 구현
def prop_function(x):
    return x/np.sum(x)


# TODO : softmax() 함수 구현
def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)    
    return x/np.sum(x)  


# 구현한 함수 확인하기
def main():
    np.random.seed(70)

    x = [1, 1, 1, 1, 6]
    
    y1 = prop_function(x)
    y2 = softmax(x)
    print("y1 = {} \ny2 = {}".format(y1, y2))



if __name__ == "__main__":
    main()




################################### [실습2] 신경망의 분류문제와 손실함수
from elice_utils import EliceUtils
import tensorflow as tf
from data.mnist import load_mnist

elice_utils = EliceUtils()

def train_model():
    # TODO : MNIST를 읽어옵니다.

    (x_train, y_train), (x_test, y_test)  = load_mnist(flatten= False, normalize = True)
    
    x_train, x_test = x_train.reshape(-1, 28, 28), x_test.reshape(-1, 28, 28)


    # TODO : 입력, 은닉, 출력을 784, 50, 10으로 합니다.
    model = tf.keras.models.Sequential([
        # (28, 28)을 (784,)로 변환 후 입력
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(50, activation='relu'),
        # 출력 10노드, softmax 사용
        tf.keras.layers.Dense(10, activation='softmax')

        ])
        
    
    # TODO : 모델을 컴파일 합니다.컴파일시 손실 함수를 `sparse_categorical_crossentropy`로 합니다.
    model.compile(optimizer='adam',
                  # 손실 함수로 sparse_categorical_crossentropy 사용
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


    # TODO : 모델을 훈련합니다.

    history = model.fit(x_train, y_train, epochs=5)
    
    # 모델의 손실값, 정확도를 측정합니다.
    model.evaluate(x_test,  y_test, verbose=2) 


    return history.history


def main():

    history = train_model()
    
    print('loss :\n', history['loss'])
    print('accuracy :\n', history['accuracy'])



if __name__ == "__main__":
    main()


##############################3 [실습3] 손실함수 실습
import numpy as np 

# TODO : 평균제곱오차 mean_square_error()함수를 구현하세요.

def mean_square_error(t, y):
    
    return np.sum((y-t)**2) / len(y)




# TODO : 교차 엔트로피 오차 함수 cross_entropy_error() 를 구현하세요.

def cross_entropy_error(t, y):
    
    return -np.sum(t*np.log(y+1e-5))



# softmax 함수입니다. 
def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x/np.sum(x)


# 구현한 함수를 통해 출력된 결과를 확인합니다.

def main():
    
    X = [[6, 4, 4, 5, 6],
        [8, 5, 9, 2, 7],
        [0, 2, 7, 9, 0],
        [6, 5, 9, 3, 8]]


    t = [[1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],

        [0, 0, 1, 0, 0]]
        
    y = []
    for i in range(len(X)):
        y.append(list(softmax(X[i])))
        
    t = np.array(t)
    y = np.array(y)
    
    mse_history = []
    for i in range(len(t)):
        mse_history.append(mean_square_error(y[i], t[i]))


    cee_history = []
    for i in range(len(t)):
        cee_history.append(cross_entropy_error(y[i], t[i]))

    print('MeanSquaredError =', mse_history)
    print('CrossEntropyError =',cee_history)


    return (mse_history, cee_history)

if __name__ == "__main__":
    main()


############################### [실습4] 경사하강법 하이퍼 파라미터 최적화
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data.mnist import load_mnist

np.random.seed(123)
tf.random.set_seed(123)

def mnist_train():
    # TODO : MNIST 데이터 세트를 읽어옵니다.


    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)
    
    
    x_train, x_test = x_train.reshape(-1, 28, 28), x_test.reshape(-1, 28, 28)
    
    # TODO : 훈련 데이터는 6만 개 중 1000개,
    # 검증 데이터는 1만 개 중 200개를 사용합니다.
    (x_train, t_train), (x_test, t_test) = (x_train[:1000], t_train[:1000]), (x_test[:200], t_test[:200]) 




    # 모델을 생성합니다.
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10,  activation='softmax')
    ])

    # TODO : 모델을 컴파일합니다. 이때 최적화방법으로 경사하강법을 사용합니다.
    # 경사하강법의 학습률을 임의의 값으로 넣어줍니다.
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.5),
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    # 훈련을 시작합니다. 5 epochs 동안 진행합니다.
    train_history = model.fit(x_train, t_train, epochs=5)
    test_eval = model.evaluate(x_test,  t_test, verbose=2)


    result = train_history.history['accuracy']
    
    return result


#####################################33 [실습5] 최적화 방법 Adam vs SGD 비교
################ main.py
# main.py
import tensorflow as tf
import numpy as np
from data.mnist import load_mnist

from sgd_model import mnist_sgd
from process import *

np.random.seed(123)

def main():


    # TODO : 경사하강법 최적 학습률을 넣어주세요.
    sgd_lr = 0.45


    hist_sgd  = mnist_sgd(sgd_lr=sgd_lr)


    hist_adam = mnist_adam()
    
    print('SGD  Accuracy : {:.4f}, loss : {:.4f}'.format(hist_sgd[-1], hist_sgd[0]))
    print('Adam Accuracy : {:.4f}, loss : {:.4f}'.format(hist_adam[-1], hist_adam[0]))
    print('Accuracy 차이 : {:.4f}, loss 차이: {:.4f}'.format(np.abs(hist_adam[-1]-hist_sgd[-1]), np.abs(hist_adam[0]-hist_sgd[0])))
    
    return sgd_lr, np.abs(hist_sgd[-1] - hist_adam[-1])



if __name__ == "__main__":
    main()


################ process.py


# process.py
 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data.mnist import load_mnist
np.random.seed(123)


def mnist_adam(epoch=5):
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)  
    x_train, x_test = x_train.reshape(-1, 28, 28), x_test.reshape(-1, 28, 28)
    (x_train, t_train), (x_test, t_test) = (x_train[:1000], t_train[:1000]), (x_test[:200], t_test[:200]) 

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10,  activation='softmax')
    ])

    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    train_history = model.fit(x_train, t_train, epochs=epoch)
    test_eval = model.evaluate(x_test,  t_test, verbose=2)
    result = test_eval
    return result


