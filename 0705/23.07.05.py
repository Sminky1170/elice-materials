################3##########[실습1] resnet 모델 사용해보기
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
import json

# 1. 이미지 전처리
# 1-1. 이미지 불러오기
img = Image.open("사진1.jpg").convert('RGB')
# 1-2. TODO: 이미지의 크기를 224x224로 변형하기
img = img.resize((224,224))
# 1-3. TODO: 이미지의 타입을 numpy array로 변환하기
np_img = np.array(img)
# 이미지를 batch로 만들기 위해 차원을 늘려주기
x = np.expand_dims(np_img, axis=0)

# 2. 모델 사용
# 2-1. TODO: imagenet로 학습된 ResNet50모델 불러오기
model = ResNet50(weights = 'imagenet')
# 2-2. TODO: 전처리된 데이터를 모델에 넣어서 예측하기
pred = model.predict(x)

# 클래스의 이름을 불러옵니다.
with open("imagenet_label_ko.json","rt",encoding='utf8') as label_file:
    classname = json.load(label_file)

# 3. 모델의 출력결과 해석하기
# 3-1. TODO: top_k함수를 이용해 결과중 가장 확률이 높은 3개만 가져오기
k = 3
topk = tf.math.top_k(pred[0], k)

# 3-2. k개 결과를 순서대로 출력하기
for i in range(k):
    # 해당 클래스의 확률을 표시하기 위해 topk.values.numpy()[i]에 100을 곱하여 score에 저장합니다.
    score = 100 * topk.values.numpy()[i]
    # 해당 클래스의 이름을 가져옵니다.
    result = classname[str(topk.indices.numpy()[i])]
    # 확률과 클래스이름을 출력합니다.
    print(f"Top-{i+1}: {result} {score:.2f}%")
#########################33[실습2] cnn 모델로 cifar-10 분류하기

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from elice_utils import EliceUtils

elice_utils = EliceUtils()

import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt

def load_cifar10():
    # CIFAR-10 데이터셋을 불러옵니다.
    X_train = np.load("cifar10_train_X.npy")
    y_train = np.load("cifar10_train_y.npy")
    X_test = np.load("cifar10_test_X.npy")
    y_test = np.load("cifar10_test_y.npy")

    # TODO: [지시사항 1번] 이미지의 각 픽셀값을 0에서 1 사이로 정규화하세요.
    X_train = X_train / 255.0

    X_test = X_test / 255.0
    
    # 정수 형태로 이루어진 라벨 데이터를 one-hot encoding으로 바꿉니다.
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)


    return X_train, X_test, y_train, y_test

def build_cnn_model(num_classes, input_shape):
    model = Sequential()

    # TODO: [지시사항 2번] 지시사항 대로 CNN 모델을 만드세요.
    model.add(Input(shape=input_shape))
    model.add(layers.Conv2D(16, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))

    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))

    model.add(layers.Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(512, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D(2))
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model

def plot_loss(hist):
    # hist 객체에서 train loss와 valid loss를 불러옵니다.
    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    epochs = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_xticks(list(epochs))
    
    # ax를 이용하여 train loss와 valid loss를 plot 합니다..
    ax.plot(epochs, train_loss, marker=".", c="blue", label="Train Loss")
    ax.plot(epochs, val_loss, marker=".", c="red", label="Valid Loss")


    ax.legend(loc="upper right")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    fig.savefig("loss.png")

def plot_accuracy(hist):
    # hist 객체에서 train accuracy와 valid accuracy를 불러옵니다..
    train_acc = hist.history["accuracy"]
    val_acc = hist.history["val_accuracy"]
    epochs = np.arange(1, len(train_acc) + 1)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xticks(list(epochs))
    # ax를 이용하여 train accuracy와와 valid accuracy와를 plot 합니다.
    ax.plot(epochs, val_acc, marker=".", c="red", label="Valid Accuracy")
    ax.plot(epochs, train_acc, marker=".", c="blue", label="Train Accuracy")

    ax.legend(loc="lower right")
    ax.grid()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    fig.savefig("accuracy.png")

def get_topk_accuracy(y_test, y_pred, k=1):
    # one-hot encoding으로 이루어진(y_test를 다시 정수 라벨 형식으로 바꿉니다.
    true_labels = np.argmax(y_test, axis=1)

    # y_pred를 확률값이 작은 것에서 큰 순서로 정렬합니다.
    pred_labels = np.argsort(y_pred, axis=1)

    correct = 0
    for true_label, pred_label in zip(true_labels, pred_labels):
        # TODO: [지시사항 3번] 현재 pred_label에서 확률값이 가장 큰 라벨 k개를 가져오세요
        cur_preds = pred_label[-k:]

        if true_label in cur_preds:
            correct += 1

    # TODO: [지시사항 3번] Top-k accuarcy를 구하세요.
    topk_accuracy = correct / len(true_labels)

    return topk_accuracy

def main(model=None, epochs=5):
    # 시드 고정을 위한 코드입니다. 수정하지 마세요!
    tf.random.set_seed(2022)

    X_train, X_test, y_train, y_test = load_cifar10()
    cnn_model = build_cnn_model(len(y_train[0]), X_train[0].shape)
    cnn_model.summary()

    # TODO: [지시사항 4번] 지시사항 대로 모델의 optimizer, loss, metrics을 설정하세요.
    optimizer = SGD(learning_rate=1e-2, momentum=0.9, nesterov=True)
    cnn_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # TODO: [지시사항 5번] 지시사항 대로 hyperparameter를 설정하여 모델을 학습하세요.
    hist = cnn_model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_split=0.2, shuffle=True, verbose=2)

    # Test 데이터를 적용했을 때 예측 확률을 구합니다.
    y_pred = cnn_model.predict(X_test)
    top1_accuracy = get_topk_accuracy(y_test, y_pred)

    top3_accuracy = get_topk_accuracy(y_test, y_pred, k=3)
    
    print("Top-1 Accuracy: {:.3f}%".format(top1_accuracy * 100))
    print("Top-3 Accuracy: {:.3f}%".format(top3_accuracy * 100))


    # Test accuracy를 구합니다.
    _, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)

    # Tensorflow로 구한 test accuracy와 top1 accuracy는 같아야 합니다.
    # 다만 부동 소수점 처리 문제로 완전히 같은 값이 나오지 않는 경우도 있어서
    # 소수점 셋째 자리까지 반올림하여 비교합니다.
    assert round(test_accuracy, 3) == round(top1_accuracy, 3)

    plot_loss(hist)
    plot_accuracy(hist)

    return optimizer, hist

if __name__ == '__main__':
    main()
###########################[실습3] data augmentation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from elice_utils import EliceUtils

elice_utils = EliceUtils()

import tensorflow as tf
from tensorflow.keras import layers, Sequential

IMG_SIZE = 256

def main():
    dog = tf.keras.utils.load_img("./dog.jpg")

    cat = tf.keras.utils.load_img("./cat.jpg")
    
    dog_array = tf.keras.utils.img_to_array(dog)
    cat_array = tf.keras.utils.img_to_array(cat)
    
    # TODO: [지시사항 1번] 개 사진에 전처리를 수행하는 모델을 완성하세요.
    dog_augmentation = Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1. / 255),
        layers.RandomCrop(150, 200)
    ])
    
    # TODO: [지시사항 2번] 고양이 사진에 전처리를 수행하는 모델을 완성하세요.
    cat_augmentation = Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1. / 255),
        layers.RandomFlip(),
        layers.RandomRotation(0.5)
    ])
    
    dog_augmented_tensor = dog_augmentation(dog_array)
    dog_augmented = tf.keras.utils.array_to_img(dog_augmented_tensor.numpy())
    dog_augmented.save("./dog_augmented.jpg")
    print("=" * 25, "전처리된 개", "=" * 25)
    elice_utils.send_image("./dog_augmented.jpg")
    
    print()
    
    cat_augmented_tensor = cat_augmentation(cat_array)
    cat_augmented = tf.keras.utils.array_to_img(cat_augmented_tensor.numpy())
    cat_augmented.save("./cat_augmented.jpg")
    print("=" * 25, "전처리된 고양이", "=" * 25)
    elice_utils.send_image("./cat_augmented.jpg")
    
    return dog_augmentation, cat_augmentation


if __name__ == "__main__":
    main()
