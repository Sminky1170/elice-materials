################################################### 1[실습1] 단어토큰 만들기

from elice_utils import EliceUtils
import codecs
import nltk
from nltk.tokenize import word_tokenize

elice_utils = EliceUtils()

# 실습 환경에 미리 설치가 됨
#nltk.download('punkt')


def count_words(input_text):
    """
    input_text 내 단어들의 개수를 세는 함수
    :param input_text: 텍스트
    :return: dictionary, key: 단어, value: input_text 내 단어 개수

    """
    
    # <ToDo>: key: 단어, value: input_text 내 단어 개수인 output_dict을 만듭니다.
    output_dict = dict()
    tokens = word_tokenize(input_text)


    for one_token in tokens:
        try:
            output_dict[one_token] += 1
        except KeyError:
            output_dict[one_token] = 1

    return output_dict


def main():
    # 데이터 파일인 'text8_1m_part_aa.txt'을 불러옵니다.
    with codecs.open("data/text8_1m_part_aa.txt", "r", "utf-8") as html_f:
        text8_text = "".join(html_f.readlines())

    # 데이터 내 단어들의 개수를 세어봅시다.

    word_dict = count_words(text8_text)
    
    # 단어 개수를 기준으로 정렬하여 상위 10개의 단어를 출력합니다.
    top_words = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    print(top_words)


    return word_dict


if __name__ == "__main__":
    main()



#################################################### [실습2] 노이즈 제거

# from elice_utils import EliceUtils
import codecs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# elice_utils = EliceUtils()
# 엘리스 환경에서 nltk data를 사용하기 위해서 필요합니다.
nltk.data.path.append("./")

def count_words(input_text):
    """
    input_text 내 단어들의 개수를 세는 함수
    :param input_text: 텍스트
    :return: dictionary, key: 단어, value: input_text 내 단어 개수
    """
    output_dict = dict()
    tokens = word_tokenize(input_text)

    for one_token in tokens:
        try:
            output_dict[one_token] += 1
        except KeyError:
            output_dict[one_token] = 1

    return output_dict


def remove_stopwords(input_dict):
    """
    input_dict 내 단어 중 stopwords 제거
    :param input_dict: count_words 함수 반환값인 dictionary
    :return: input_dict에서 stopwords가 제거된 것
    """
    stop_words = set(stopwords.words('english'))

    output_dict = dict()
    for one_word, one_value in input_dict.items():
        if one_word not in stop_words:
            output_dict[one_word] = one_value

    return output_dict


def remove_less_freq(input_dict, lower_bound=10):
    """
    input_dict 내 단어 중 lower_bound 이상 나타난 단어만 추출
    :param input_dict: count_words 함수 반환값인 dictionary
    :param lower_bound: 단어를 제거하는 기준값
    :return: input_dict에서 lower_bound이상 나타난 단어들
    """
    output_dict = dict()
    for one_word, one_value in input_dict.items():
        if one_value >= lower_bound:
            output_dict[one_word] = one_value

    return output_dict


def main():
    with codecs.open("data/text8_1m_part_aa.txt", "r", "utf-8") as html_f:
        text8_text = "".join(html_f.readlines())

    word_dict1 = count_words(text8_text)
    word_dict2 = remove_stopwords(word_dict1)
    word_dict3 = remove_less_freq(word_dict2)

    print("# word_dict1: {}".format(len(word_dict1)))
    print("# word_dict2: {}".format(len(word_dict2)))
    print("# word_dict3: {}".format(len(word_dict3)))

    top_words1 = sorted(word_dict1.items(), key=lambda x: x[1], reverse=True)[:15]
    print("word_dict1 topwords: {}".format(top_words1))
    top_words2 = sorted(word_dict2.items(), key=lambda x: x[1], reverse=True)[:15]
    print("word_dict2 topwords: {}".format(top_words2))
    top_words3 = sorted(word_dict3.items(), key=lambda x: x[1], reverse=True)[:15]
    print("word_dict3 topwords: {}".format(top_words3))

    return word_dict3


if __name__ == "__main__":
    main()


#################################################### [실습3] 단어임베딩과 원핫인코딩

import response
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer


sentence1 = ['나','는','오늘','저녁','에','치킨','을','먹','을','예정','입니다']
sentence2 = ['나','는','어제', '맥주','와', '함께', '치킨','을', '먹었', '습니다']


def main():
    tokenizer=Tokenizer()

    # TODO: Tokenizer를 Text(sentence1, sentence2)에 맞추고 단어 인덱스를 word_dict에 저장합니다.
    tokenizer.fit_on_texts(sentence1+sentence2)
    word_dict = tokenizer.word_index

    # TODO: Tokenizer를 사용하여 sentence1, sentence2 를 정수값으로 변환하고 이를 시퀀스로 반환합니다.
    sen1 = tokenizer.texts_to_sequences(sentence1)
    sen2 = tokenizer.texts_to_sequences(sentence2)

    sen1 = [ token[0] for token in sen1]
    sen2 = [ token[0] for token in sen2]

    # TODO: Tensorflow를 사용하여 원-핫 인코딩을 실행합니다.(원-핫 벡터의 총길이는 word_dict 안의 word의 총 개수). 단어를 원-핫 인코딩 후 이것을 문장별로 (요소별) 더함.
    oh_sen1 = sum(tf.one_hot(sen1, len(word_dict)))
    oh_sen2 = sum(tf.one_hot(sen2, len(word_dict)))

    print("원-핫 인코딩된 문장1:", oh_sen1.numpy())
    print("원-핫 인코딩된 문장2:", oh_sen2.numpy())

    # TODO: 원-핫 벡터를 바탕으로 코사인 유사도를 구합니다.
    cos_simil = sum(list(oh_sen1*oh_sen2)) / (tf.norm(oh_sen1)*tf.norm(oh_sen2))
    print("두 문장의 코사인 유사도:", cos_simil.numpy())

    # TODO: 원-핫 벡터의 길이(차원)를 확장시킨 후 코사인 유사도를 구하여, 이전의 값과 비교해 봅니다.
    len_word=500000

    oh_sen1 = sum(tf.one_hot(sen1, len_word))
    oh_sen2 = sum(tf.one_hot(sen2, len_word))

    cos_simil = sum(list(oh_sen1*oh_sen2)) / (tf.norm(oh_sen1)*tf.norm(oh_sen2))
    
    print("원-핫 인코딩의 길이가 500,000일 때의 코사인 유사도:", cos_simil.numpy())


if __name__ == '__main__':
    main()




##################################################### [실습4] Word2Vec
# from elice_utils import EliceUtils
import codecs
import gensim
from gensim.models.word2vec import Word2Vec
# elice_utils = EliceUtils()


def compute_similarity(model, word1, word2):
    """
    word1과 word2의 similarity를 구하는 함수
    :param model: word2vec model
    :param word1: 첫 번째 단어
    :param word2: 두 번째 단어
    :return: model에 따른 word1과 word2의 cosine similarity
    """
    similarity = model.wv.similarity(word1, word2)

    return similarity


def get_word_by_calculation(model, word1, word2, word3):
    """
    단어 벡터들의 연산 결과 추론하는 함수
    연산: word1 - word2 + word3
    :param model: word2vec model
    :param word1: 첫 번째 단어로 연산의 시작
    :param word2: 두 번째 단어로 빼고픈 단어
    :param word3: 세 번째 단어로 더하고픈 단어
    :return: 벡터 계산 결과에 가장 알맞는 단어
    """
    output_word = model.wv.most_similar(positive=[word1, word3], negative=[word2])[0][0]

    return output_word


def main():
    model = Word2Vec.load('./data/w2v_model')

    word1 = "이순신"
    word2 = "원균"
    word1_word2_sim = compute_similarity(model, word1, word2)
    print("{}와/과 {} 유사도: {}".format(word1, word2, word1_word2_sim))

    word1 = "대한민국"
    word2 = "서울"
    word3 = "런던"
    cal_result = get_word_by_calculation(model, word1, word2, word3)
    print("{} - {} + {}: {}".format(word1, word2, word3, cal_result))

    word1 = "세종"
    word2 = "태종"
    word1_word2_sim = compute_similarity(model, word1, word2)
    print("{}와/과 {} 유사도: {}".format(word1, word2, word1_word2_sim))

    word1 = "교수"
    word2 = "학교"
    word3 = "학생"
    cal_result = get_word_by_calculation(model, word1, word2, word3)
    print("{} - {} + {}: {}".format(word1, word2, word3, cal_result))

    return word1_word2_sim, cal_result


if __name__ == "__main__":
    main()


#################################################### [실습5] 순환신경망(RNN)
import numpy as np

def rnn(inputs, input_size, output_size, bias = True):
    # TODO: 0의 값을 갖는 (output_size,) 모양의 state 벡터를 만들어 봅니다.
    state = np.zeros((output_size,))
    # TODO: 1의 값을 갖는 (output_size, input_size) 모양의 w 벡터를 만들어 봅니다.
    w = np.ones((output_size, input_size))
    # TODO: 1의 값을 갖는 (output_size, output_size) 모양의 u벡터를 만들어 봅니다.
    u = np.ones((output_size, output_size))
    # TODO: 임의의 값을 갖는 (output_size,) 모양의 b벡터를 만들어 봅니다.

    b = np.random.random((output_size,))
    
    # TODO: bias 가 False 이면 b를 (output_size,) 모양의 영벡터를 만들어 줍니다.
    if not bias:
        b=np.zeros((output_size,))
        
    outputs = []
    
    for _input in inputs:
        # TODO: (Numpy 사용) w와 _input을 내적하고, u 와 state를 내적한 후 b를 더한 다음 하이퍼볼릭 탄젠트 함수를 적용합니다.
        _output = np.tanh(np.dot(w, _input) + np.dot(u, state) + b)
        outputs.append(_output)
        state=_output
        
    return np.stack(outputs, axis=0) 



## TODO: 입력과 출력을 바꾸어 가며 RNN의 원리를 파악해 봅니다.

input_1 = [0, 0, 0, 0, 0]
input_2 = [1, 1, 1, 1, 1]
input_3 = [1, 2, 3, 4, 5]
input_4 = [5, 4, 3, 2, 1]


print(rnn(input_1, input_size=1, output_size=1))
print(rnn(input_1, input_size=1, output_size=1, bias = False))

print(rnn(input_2, input_size=5, output_size=5))
print(rnn(input_2, input_size=5, output_size=5, bias = False))

print(rnn(input_3, input_size=5, output_size=5))
print(rnn(input_3, input_size=5, output_size=5, bias = False))

print(rnn(input_4, input_size=5, output_size=5))
print(rnn(input_4, input_size=5, output_size=5, bias = False))


#################################################### [실습6] Simple RNN 실습
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras import Model
import tensorflow as tf

from elice_utils import EliceUtils

from utils import drawRNNLoss, calc_in4p, read_sequence
elice_utils = EliceUtils()

def main():
    # 초기화 : 아래 seed 값은 수정하지 마세요.
    tf.random.set_seed(90)

    np.random.seed(90)
    
    # TODO : 데이터 읽어들이기
    (x_train, t_train), (x_test, t_test) = read_sequence()
    
    
    # 전체 학습이 진행되기 전 설정값 확인에 사용합니다.
    data_size = 100
    (x_train, t_train), (x_test, t_test) = (x_train[:data_size], t_train[:data_size]), (x_test[:data_size], t_test[:data_size])
    
    
    # TODO : SimpleRNN 신경망 구현
    srnn_100_model = tf.keras.Sequential([
            tf.keras.layers.SimpleRNN(units=30, return_sequences=True, input_shape=[100,2]),
            tf.keras.layers.SimpleRNN(units=30), 
            tf.keras.layers.Dense(1)
            ])
    
    
    # TODO : optimizer, loss 설정합니다.
    srnn_100_model.compile(optimizer='adam', loss='mse') 
    
    
    # TODO : epoch와 verbose 설정
    srnn_100_history = srnn_100_model.fit(x_train, t_train, epochs=100, validation_split=0.2, verbose=2)
    
    
    # 그래프 확인
    drawRNNLoss(srnn_100_history)
    
    
    # 라벨에 있는 값과 예측한 값의 차이가 0.1 이내면 맞은 것으로 처리
    prediction = srnn_100_model.predict(x_test)
    calc_in4p(prediction, 0.1)
    
    return prediction


if __name__ == "__main__":
    main()

    