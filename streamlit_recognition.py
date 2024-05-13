import streamlit as st #pip install streamlit
import cv2
import mediapipe as mp #pip install mediapipe
import os
import numpy as np
import tensorflow as tf #pip install tensorflow==2.16.1
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers import Dense


#Инициализация объектов mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic()

HAND_CONNECTIONS = mp_holistic.HAND_CONNECTIONS
POSE_CONNECTIONS = mp_holistic.POSE_CONNECTIONS
FACE_CONNECTIONS = mp_holistic.FACEMESH_CONTOURS

#порядок отрисовки точек
drawing_spec1 = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3) #color не rgb а bgr в этом методе
drawing_spec2 = mp_drawing.DrawingSpec(color=(100, 200, 50), thickness=2, circle_radius=1)

def draw_hands(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec = drawing_spec1)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,landmark_drawing_spec = drawing_spec1)


def draw_face(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec = drawing_spec2 )


def draw_pose(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

#сбор ключевых точек
def get_points(results):
    NUM_POSE_KEYPOINTS = 33*4 #33 точки * (xyz + visibility)
    NUM_FACE_KEYPOINTS = 468 * 3
    NUM_HAND_KEYPOINTS = 21 * 3
    #TOTAL 1662
    #если в кадре были все точки(лицо, поза, и обе руки, то мы получим все 1662 координаты, иначе заполним нулями, где не получили)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(NUM_POSE_KEYPOINTS)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(NUM_FACE_KEYPOINTS)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(NUM_HAND_KEYPOINTS)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(NUM_HAND_KEYPOINTS)
    return np.concatenate([pose, face, lh, rh])


#############################Часть инициализации модели ################################################

gestures = np.array(['salemetsiz be', 'men', 'ymtardy', 'uirenyp jatyrmyn'])#группа 1
#gestures = np.array(['bugin', 'aua raiy', 'keremet', 'ystyq']) #группа 2
#gestures = np.array(['atesh', 'jaman', 'doreki', 'bileu'])#группа 3
#gestures = np.array(['tynysh', 'suyq', 'ushaq', 'jaksy'])#группа 4
#gestures = np.array(['rakhmet', 'sau bolynyz', 'keude', 'telefon'])#группа 5

model = Sequential()
#в нс подается 30 кадров по 1662 точки
#добавляем первый LSTM слой, состоит из 64 нейронов и возвращает последовательность
# Это означает, что выходной тензор этого слоя будет иметь такую же размерность, как и его входной тензор.
#return_sequences=True мы указываем только в том случае,если след слой тоже lstm
#на выходе мы должны классифицировать столько меток класса, сколько у нас есть в gestures слов
#нам модель вернет вероятность для каждого слова, например hello = 0.9, iloveyo - 0.2
#Мы берем то слово, у которого большая вероятность
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(gestures.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.load_weights('final_weights/weights_group_1.h5')
#model.load_weights('final_weights/weights_group_2_v2.h5')
#model.load_weights('final_weights/weights_group_3.h5')
#model.load_weights('final_weights/weights_group_4_v2.h5')
#model.load_weights('final_weights/weights_group_5.h5')


def recognize_gestures():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8
    previous_gesture = None
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Create an empty placeholder to display the video stream
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error reading frame")
            break
        res = holistic.process(frame)
        if res.face_landmarks:
            draw_face(frame, res)
        if res.left_hand_landmarks or res.right_hand_landmarks:
            draw_hands(frame, res)
        if res.pose_landmarks:
            draw_pose(frame, res)
        keypoints = get_points(res)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        if len(sequence) == 30:
            answ = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(answ))
            if answ[np.argmax(answ)] > threshold:
                current_gesture = gestures[np.argmax(answ)]
                if current_gesture != previous_gesture:
                    sentence = [current_gesture]
                    previous_gesture = current_gesture
        cv2.rectangle(frame, (0, 0), (640, 40), (245, 117, 16), -1)
        if res.left_hand_landmarks or res.right_hand_landmarks:
            cv2.putText(frame, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", width=740)

def main():

    # Добавляем навигацию по страницам в боковой панели
    st.set_page_config(page_title="Gesture Recognition Demo",
                       page_icon=":tada:",
                       layout="wide",
                       initial_sidebar_state="expanded")
    page = st.sidebar.selectbox("Выберите страницу", ["Главная", "О нейронной сети", "Демо"])

    page_bg_img = f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
        background-image: url("https://i.postimg.cc/wjMFWJ6c/background-font.jpg");
        background-size: cover;
        background-position: center center;
        background-repeat: no-repeat;
        background-attachment: local;
        }}
        [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
        }}
        </style>
        """

    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.markdown(page_bg_img, unsafe_allow_html=True)
    # Отображаем содержимое выбранной страницы
    if page == "Главная":
        show_home_page()
    elif page == "О нейронной сети":
        show_about_page()
    elif page == "Демо":
        show_contact_page()

def show_home_page():

    st.title("Алгоритм распознавания жестов в реальном времени")
    st.write("<p style='font-size: 22px; '>В данном проекте реализован функционал по распознаванию динамических жестов при помощи "
             "современных методов компьютерного зрения. За детекцию отдельного жеста и его классификацию отвечает "
             "модель машинного обучения на основе нейронной сети, которая может быть интегрирована в различные сервисы и приложения. Модель разработана "
             "с применением фреймворка <b>Tensorflow</b>, который позволяет с минимальными задержками отдавать предсказания пользователю. "
             "<br><br>", unsafe_allow_html=True)

    # Создаем два столбца
    col1, col2 = st.columns([2, 3])

    with col1:
        st.image("./web/pic1.jpeg",  use_column_width=False, width=440)

    with col2:
        st.write("""
            <div>
                <p style='font-size: 22px;'>Все, что необходимо для работы нейронной сети - это <b>камера на вашем устройстве</b>. После ее активации
             модель сможет распознать <b>30 жестов</b>, на которых предварительно была обучена. Нейронная сеть выдает предсказания, только если уверенность достигает более 
         <b>80%. </b> 
         Во вкладке "Демо" пользователь может воспользоваться функцией по распознаванию жестов, 
         предварительно рекомендется ознакомиться с инструкцией
         </p>
            </div>
            """, unsafe_allow_html=True)

def show_about_page():
    st.title("О нейронной сети")
    st.header("Как собирались тренировочные данные?")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.write("""
                    <div>
                        <p style='font-size: 22px;'>Условно жесты можно разделить на <b>статичные(без активной динамики кистей рук)</b> и <b>динамические</b>.
                        В случае со статичными жестами, было бы достаточно обучать модель на наборе фотографий, так как положение рук не изменяется.
                        Но при работе с жестами, где присутствует активная динамика кистей рук такой подход является неверным.
                        <br></br>
                        </p>
                        <p style='font-size: 22px;'>
                        Для данного проекта данные собирались вручную следующим образом: 
                            <ol style='font-size: 22px;'>
                              <li style='font-size: 22px;'>Каждый жест показывается на камеру <b>30</b> раз и записывается в виде <b>30</b> последовательностей</li>
                              <li style='font-size: 22px;'>Для каждой последовательности записывается <b>40 кадров</b>, и помощью библиотеки <b>Mediapipe</b> с каждого кадра снимаются координаты точек рук</li>
                              <li style='font-size: 22px;'>Координаты сохраняются в отдельные файлы и подаются на вход нейронной сети для обучения</li>
                              <li style='font-size: 22px;'>Для real-time предсказания с видеопотока забирается по 40 кадров и выдатся предсказание</li>
                            </ol>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    with col2:
        st.image("./web/my_pic.png", use_column_width=False, width=500)

    st.header("В чем особенность архитектуры нейронной сети?")
    col1, col2 = st.columns([2, 3])

    with col1:
        st.image("./web/LSTM.png", use_column_width=False, width=450)

    with col2:
        st.write("""
                    <div>
                        <p style='font-size: 22px;'>В данной задаче жест - последовательность данных, соответсвенно 
                        нейронная сеть должна обладать некоторой <b>"памятью"</b>, чтобы связывать отдельные движения рук в общий жест.
                        Для этого оптимально использовать <b>LSTM (Long-Short-Term-Memory)</b> слои, способные сохранять информацию о предыдущих состояниях 
                        и использовать ее при принятии решений на последующих шагах. 
                        <br> </br>
                        Также LSTM слои предлагают решение проблемы затухания градиента, поскольку они имеют механизмы контроля потока информации через <b>gates(ворота)</b>, что позволяет более эффективно передавать градиенты на более длинные расстояния
                        <br></br>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
def show_contact_page():
    st.title("Начните распознавание языка жестов")
    st.header("Общие рекомендации")
    st.write("""<ol style='font-size: 22px;'>
                    <li style='font-size: 22px;'>Обеспечьте хорошее освещение в помещении</li>
                    <li style='font-size: 22px;'>Сядьте на расстоянии вытянутой руки от камеры, чтобы захватить координаты точек позы</li>
                    <li style='font-size: 22px;'>При показе жестов убедитесь, что кисти рук выделяляются синим цветом и видны нейронной сети</li>
                    <li style='font-size: 22px;'>Не показывайте жесты слишком быстро</li>
             </ol>""", unsafe_allow_html=True)

    # при нажатии 1 раз, откроется камера с распознаванием, после можно нажать для ее отключения
    # хранение состояния кнопки необходимо для последующего отключения камеры
    # при желании можно сменить

    session_state = st.session_state
    if "camera_started" not in session_state:
        session_state["camera_started"] = False

    if st.button("Начать/ закончить распознавание"):
        session_state["camera_started"] = not session_state["camera_started"]
        if session_state["camera_started"]:
            recognize_gestures()


if __name__ == "__main__":
    main()