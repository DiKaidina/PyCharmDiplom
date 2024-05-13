import numpy as np
import os
import os
import numpy as np
#from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from tensorflow.python.keras.regularizers import l2


N_VIDEOS = 30
N_FRAMES = 30

def label_map(gestures):
    INPUT_DIR = "/kaggle/input/gesture-group-1/gesture_group_1"
    label_map = {label: num for num, label in enumerate(gestures)}
    sequences, labels = [], []
    for gesture in gestures:
        for sequence in np.array(os.listdir(os.path.join(INPUT_DIR, gesture))):
            window = []  # промежуточный массив - собирает кадры для конкретной последовательности
            for frame_num in range(N_VIDEOS):
                res = np.load(os.path.join(INPUT_DIR, gesture, str(sequence), "frame_{}.npy".format(frame_num)))
                window.append(res)  # в цикле собрали все кадры для одной последовательности
            sequences.append(window)  # сунули эту последовательность в другой массив (итого массив в массиве)
            labels.append(label_map[gesture])  # проставили метку класса

'''
def train_test_split():
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

'''

def config_model(gestures):
    model = Sequential()

    # Добавление LSTM слоев с регуляризацией и Dropout
    model.add(LSTM(128, return_sequences=True, activation='relu', kernel_regularizer=l2(0.01), input_shape=(30, 1662)))
    model.add(Dropout(0.4))

    model.add(LSTM(256, return_sequences=True, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))

    model.add(LSTM(128, return_sequences=False, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.4))

    # Добавление полносвязных слоев с регуляризацией
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.01)))

    # Выходной слой с функцией активации softmax
    model.add(Dense(gestures.shape[0], activation='softmax'))

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])



def main():
    gestures = np.array(['salemetsiz be', 'men', 'ymtardy', 'uirenyp jatyrmyn'])

    label_map(gestures)

    config_model(gestures)

#обучение модели
'''
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    history = model.fit(X_train, y_train, epochs=8000)
'''
