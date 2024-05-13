import cv2
import mediapipe as mp
import os
import numpy as np

def create_directories_from_array(array):
    for item in array:
        # Создание основной папки для текущего элемента
        directory_path = os.path.join(OUTPUT_DIR, item)
        create_directory_if_not_exists(directory_path)

        # Создание подпапок seq_1, seq_2, ..., seq_100
        for i in range(1, N_VIDEOS + 1):
            seq_directory_path = os.path.join(directory_path, f'seq_{i}')
            create_directory_if_not_exists(seq_directory_path)


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Папка {directory} успешно создана")
    else:
        print(f"Папка {directory} уже существует")


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

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic()

#корневая папка и гиперпараметры
OUTPUT_DIR = "C:/Users/DIANA/Desktop/Diploma Work/gesture_landmarks_test"
N_VIDEOS = 30
N_FRAMES = 30

gestures = np.array(['salemetsiz be', 'men', 'ymtardy', 'uirenyp jatyrmyn'])#группа 1
#gestures = np.array(['bugin', 'aua raiy', 'keremet', 'ystyq']) #группа 2
#gestures = np.array(['atesh', 'jaman', 'doreki', 'bileu'])#группа 3
#gestures = np.array(['tynysh', 'suyq', 'ushaq', 'jaksy'])#группа 4
#gestures = np.array(['telefon', 'rakhmet', 'keude', 'sau bolynyz'])#группа 5


HAND_CONNECTIONS = mp_holistic.HAND_CONNECTIONS
POSE_CONNECTIONS = mp_holistic.POSE_CONNECTIONS
FACE_CONNECTIONS = mp_holistic.FACEMESH_CONTOURS


drawing_spec1 = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3) #color не rgb а bgr в этом методе
drawing_spec2 = mp_drawing.DrawingSpec(color=(100, 200, 50), thickness=2, circle_radius=1)

def draw_hands(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec = drawing_spec1)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,landmark_drawing_spec = drawing_spec1)
def draw_face(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec = drawing_spec2 )
def draw_pose(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def main():

    create_directories_from_array(gestures)

    cap = cv2.VideoCapture(0)

    # надо менять каждый раз при показе нового жеста
    VIDEO_FOLDER = os.path.join(OUTPUT_DIR, 'salemetsiz be')

    for video in range(1, N_VIDEOS + 1):
        # путь к записи последовательности
        seq_folder = os.path.join(VIDEO_FOLDER, f"seq_{video}")
        for img in range(N_FRAMES):
            ret, frame = cap.read()
            if not ret:
                print("Ошибка при чтении кадра")
                break

            res = holistic.process(
                frame)  # модель анализирует содержимое кадра и обнаруживает ключевые точки лица, рук и тела

            if res.face_landmarks:
                draw_face(frame, res)

            if res.left_hand_landmarks or res.right_hand_landmarks:
                draw_hands(frame, res)

            if res.pose_landmarks:
                draw_pose(frame, res)

            # сигнал к началу  записи(если это первый кадр последовательности)

            if img == 0:
                ##отдельная функция(перед сдачей)
                cv2.putText(frame, 'Started collecting', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f'Starting seq_{video}', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)
                cv2.imshow('data', frame)
                cv2.waitKey(1000)  # чтобы было время подождать до записи нового кадра
            else:
                cv2.putText(frame, f'Collecting frames for seq_{video}', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('data', frame)

            last_keypoints = get_points(res)
            np.save(os.path.join(seq_folder, f"frame_{img}.npy"), last_keypoints)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
