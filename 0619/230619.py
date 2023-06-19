###################### 퍼셉트론으로 NOR 함수 만들기
from score import scoring


def NOR(x1, x2):
    # w1, w2, q 를 저장해 둘 빈 리스트
    # TODO - w1, w2, q 값을 넣어주세요.

    nor_wq = [-0.4, -0.4, -0.3]
    
    w1, w2, q = nor_wq[0], nor_wq[1], nor_wq[2]
    if (x1*w1 + x2*w2 <= q) :
        return 0
    else :
        return 1



def main():
    # 정답과 작성한 코드의 결과를 확인하는 함수입니다.
    scoring()


if __name__ == "__main__":
    main()


######################## 퍼셉트론으로 IRIS 데이터 이진분류
import numpy as np
import pandas as pd

from graph_plot import *

from elice_utils import EliceUtils

elice_utils = EliceUtils()

# TODO : iris 퍼셉트론 분류 함수를 위한 w1, w2, q 설정
wq = [-2.0, 11.0, 16.0]


def pre_data():
    
    # TODO : iris 데이터를 가져옵니다.
    iris = sns.load_dataset("iris")
    
    # TODO : iris 데이터 중 'virginica' 종을 제외한 
    # 'setosa' 와 'versicolor' 를 별도로 저장합니다.
    iris_two_species = iris[iris['species'] != 'virginica'] 
    
    x = pd.DataFrame()
    x['sepal_length'] = iris_two_species['sepal_length']
    x['petal_length'] = iris_two_species['petal_length']
    x['species'] = iris_two_species['species']
    
    x1 = x[x['species']=='setosa']
    x2 = x[x['species']=='versicolor']
    
    return x1, x2
    
    
def main():
    
    # TODO : pre_data() 함수를 활용하여 데이터를 불러옵니다.
    x1, x2 = pre_data()
    
     # 'setosa' 와 'versicolor' 종의 꽃잎과 꽂받침 길이를 시각화합니다.
    data_plot(x1, x2)
    
    result = []
    
    # 꽃잎과 꽃받침의 길이를 기준으로 
    # 0.1 단위로 iris_div() 호출
    for i in range(43, 71, 1):
        for j in range(10, 52, 1):
            result.append([i/10.0, j/10.0, iris_div(i/10, j/10, wq)])


    # x는 sepal_length, 
    # y는 petal_length, 
    # z는 [0 : setosa, 1 : versicolor] 입니다.
    x = [i[0] for i in result]
    y = [i[1] for i in result]
    z = [i[2] for i in result]

    # iris 퍼셉트론 선형분류 시각화
    lx = min(min(x1['sepal_length']), min(x2['sepal_length']))

    rx = max(max(x1['sepal_length']), max(x2['sepal_length']))
    
    if (wq[1] < 0.0001 and wq[1] > -0.0001):
        wq[1] = 0.001
    ly = -wq[0]/wq[1]*lx + wq[2]/wq[1]
    ry = -wq[0]/wq[1]*rx + wq[2]/wq[1]


    # 산점도 및 직선 확인
    sub_plot(x, y, z, x1, x2, lx, ly, rx, ry)


# iris 데이터를 분류하는 퍼셉트론 코드
def iris_div(x1, x2, wq):

    w1, w2, q = wq[0], wq[1], wq[2] 
    if (x1*w1 + x2*w2 <= q) :
        return 0
    else :
        return 1


if __name__ == "__main__":
    main()

###########################3 다층퍼셉트론과 XOR
import numpy as np
import matplotlib.pyplot as plt
from elice_utils import EliceUtils

elice_utils = EliceUtils()

# TODO : weight 와 bias 값을 설정.
# XOR 문제를 풀기 위한 W 와 B 값 
W1 = [[-1.0, -0.5, 1.1],   # -1.0x - 0.5y + 1.1 = 0 ---(1)
      [0.7, 1.4, -0.5],    #  0.7x + 1.4y - 0.5 = 0 ---(2)
      [0, 0, 1]]

W2 = [[0.4, 0.4, -0.7],     # 0.4x + 0.4y - 0.7 = 0 ---(3)
      [0, 0, 0],
      [0, 0, 0]]

def main():

    # (0,0), (0,1), (1,0), (1,1) 을 입력으로 처리.
    Ex00 = [0,0,1]
    Ex01 = [0,1,1]
    Ex10 = [1,0,1]

    Ex11 = [1,1,1]
    
    print("x1, x2 = (0, 0) : ", DLP_XOR(Ex00,W1,W2))
    print("x1, x2 = (0, 1) : ", DLP_XOR(Ex01,W1,W2))
    print("x1, x2 = (1, 0) : ", DLP_XOR(Ex10,W1,W2))
    print("x1, x2 = (1, 1) : ", DLP_XOR(Ex11,W1,W2))
    
    result = []


    for i in range(-5, 15, 1):
        for j in range(-5, 15, 1):
            result.append([i/10.0, j/10.0, 
                DLP_XOR([i/10.0, j/10.0, 1],W1,W2)])

    # result 의 x,y, color 을 분리
    x = [i[0] for i in result]
    y = [i[1] for i in result]
    z = [i[2] for i in result]


    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(x,y,c=z, alpha=0.5)
    ax.scatter([0,1],[1,0], s=200, c='green', alpha=0.5)
    ax.scatter([0,1],[0,1], s=200, c='red', alpha=0.5)
    
    x_1 = np.array(range(-5, 15, 1))/10.0
    y_1 = -W1[0][0]/W1[0][1]*x_1 - W1[0][2]/W1[0][1]
    x_2 = np.array(range(-5, 15, 1))/10.0
    y_2 = -W1[1][0]/W1[1][1]*x_1 - W1[1][2]/W1[1][1]


    ax.plot(x_1,y_1,x_2,y_2)
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

    fig.savefig("plot.png")
    elice_utils.send_image("plot.png")


def DLP_XOR(X, W1, W2):
    # 입력
    x = np.array(X)

    # W,B 입력
    w1 = np.array(W1)    
    w2 = np.array(W2)    

    # 연산
    h1 = np.array([0.0]*len(X))
    y = np.array([0]*len(X))

    for count in range(len(X)):
        h1[count] = np.sum(x*w1[count]) > 0
    for count in range(len(x)):
        y[count] = np.sum(h1*w2[count]) > 0
    return y[0]


if __name__ == "__main__":
    main()


################################# IRIS 분류
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from elice_utils import EliceUtils
from absl import logging

logging._warn_preinit_stderr = 0
logging.warning('Worrying Stuff')

elice_utils = EliceUtils()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# iris 데이터셋 읽기
def readIris():
    # TODO : data 폴더에 위치한 iris 데이터셋을 불러오세요.

    iris = pd.read_csv('data/iris.csv')
    
    
    return iris  



# iris 데이터셋 분류 - train/test
def makeXY(iris):
    X = iris.iloc[:,1:5].values
    _y = iris.iloc[:,5].values
    _y = LabelEncoder().fit_transform(_y)
    Y = pd.get_dummies(_y).values
    return train_test_split(X, Y, test_size=0.2, random_state=25)


# model 작성
def makeModel():
    model = tf.keras.models.Sequential([
        # TODO : 은닉 노드 개수, 활성 함수 이름, 입력 노드 개수 설정 
        tf.keras.layers.Dense(32, activation='sigmoid', input_shape=(4,)),
        # 출력 노드의 개수 입력
        # TODO : 출력 노드 개수 설정
        tf.keras.layers.Dense(3, activation='softmax')])
    model.compile(loss='categorical_crossentropy', 
                optimizer='Adam', 
                metrics=['accuracy'])
    return model


# model 적중률 측정
# TODO : 검증 데이터(test)를 넣고 적중률(accuracy)을 측정하세요.
def irisEvaluate(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    return accuracy


def main():
    iris = readIris()
    X_train, X_test, y_train, y_test = makeXY(iris)

    model = makeModel()
    
    # 학습 진행 (100회, epochs=100)
    # TODO : epochs=100
    hist = model.fit(X_train, 
                     y_train, 
                     validation_data=(X_test,y_test),
                     epochs=100)



    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(hist.history['accuracy'])
    ax.plot(hist.history['val_accuracy'])
    plt.legend(['accuracy','val_accuracy'])


    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    fig.savefig("plot.png")
    elice_utils.send_image("plot.png")

    accuracy = irisEvaluate(model, X_test, y_test)
    print("Accuracy = {:.4f}".format(accuracy))
    result = {'hist': hist, 'accuracy':accuracy}
    return result


if __name__ == "__main__":
    main()

###############################3 활성함수의 종류
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.layers import Dense
from elice_utils import EliceUtils
from absl import logging

logging._warn_preinit_stderr = 0
logging.warning('Worrying Stuff')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

elice_utils = EliceUtils()


# TODO : 미분 함수 작성
def numerical_diff(f, x):

    h = 1e-5
    
    # TODO : 미분 함수의 반환값
    return  (f(x+h)-f(x-h))/(2*h)



# 활성 함수
# TODO : 아래 활성화 함수들을 만들어 봅시다..
def sigmoid(x):
    return tf.keras.activations.sigmoid(x)

def tanh(x):
    return tf.keras.activations.tanh(x)

def relu(x):
    return tf.keras.activations.relu(x)

def elu(x):
    return tf.keras.activations.elu(x)

def selu(x):
    return tf.keras.activations.selu(x)


def exec_grad(graph):
    x = np.arange(-10, 10, 0.1) 

    # 기울기 계산
    # TODO : relu 와 tanh 활성 함수의 기울기를 구해봅시다.
    y_sigmoid_grad = numerical_diff(sigmoid, x)
    y_tanh_grad = numerical_diff(tanh, x)
    y_relu_grad = numerical_diff(relu, x)
    y_elu_grad = numerical_diff(elu, x)
    y_selu_grad = numerical_diff(selu, x)

    result_grad = {'x' : x, 
                   'sigmoid' : y_sigmoid_grad, 
                   'tanh' : y_tanh_grad,
                   'relu' : y_relu_grad,
                   'elu' : y_elu_grad,
                   'selu' : y_selu_grad}


    fig, ax = plt.subplots(figsize=(10, 6))
    
    def drawGraph(x, y, title, plotNum, ax=ax):
        plt.subplot(plotNum)
        ax.set_xlim([-10.0, 10.0])
        ax.set_ylim([-0.1, 2.0])
        plt.axvline(x=0, color='r', linestyle=':')
        plt.axhline(y=0, color='r', linestyle=':')
        plt.title(title)
        plt.grid(dashes=(3,3),linewidth=0.5)
        plt.plot(x,y)
    
    
    if graph==1 :
        # 활성 함수의 기울기를 그래프로 표현해 봅시다.
        drawGraph(x, y_sigmoid_grad, 'Sigmoid', 231)
        drawGraph(x, y_tanh_grad, 'tanh', 232)    # tanh 기울기 그래프
        drawGraph(x, y_relu_grad, 'ReLU', 234)    # ReLU 기울기 그래프
        drawGraph(x, y_elu_grad, 'ELU', 235)
        drawGraph(x, y_selu_grad, 'SELU', 236)
        fig.savefig("iris_plot.png")
        elice_utils.send_image("iris_plot.png")



    # x = (-9, -5, -1, -0.2, 0.2, 1, 5, 9) 에서의 각 활성 함수의 기울기
    # index = (10, 50, 90, 98, 102, 110, 150, 190) 
    print('     Gradient of ...\n   X | Sigmoid |   tanh  |   ReLU  |  SELU ')
    for i in [10, 50, 90, 98, 102, 110, 150, 190]:
        tf.print('{:4.1f} |{:.6f} |{:.6f} |{:.6f} |{:.6f}'.format(x[i] ,
                    result_grad['sigmoid'][i],
                    result_grad['tanh'][i],
                    result_grad['relu'][i],
                    result_grad['selu'][i]))

    return result_grad


if __name__ == "__main__":
    exec_grad(graph=1)