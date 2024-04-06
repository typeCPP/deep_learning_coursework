import numpy as np
import pandas as pd
import cv2
import time
import matplotlib.pyplot as plt
import pickle

from keras.models import load_model

# Чтение csv файла с классами знаков
labels = pd.read_csv('input/traffic-signs-preprocessed/label_names.csv')

model = load_model('model-3x3.h5')

# Загрузка среднего изображения для дальнейшей предобработки
# Открытие файла для чтения в бинарном режиме
with open('input/traffic-signs-preprocessed/mean_image_rgb.pickle', 'rb') as total_frames:
    mean = pickle.load(total_frames, encoding='latin1')  # dict

# Обученные веса YOLO
path_to_cfg = 'input/traffic-signs-dataset-in-yolo-format/yolov3_ts_test.cfg'
path_to_weights = 'input/car_data/znaki_rtx_final.weights'

# Загрузка обученных весов YOLO и конфигурационного файла с помощью OpenCV
network = cv2.dnn.readNetFromDarknet(path_to_cfg, path_to_weights)

# Получение всех слоев YOLO
layers_all = network.getLayerNames()
layers_names_output = [layers_all[i - 1] for i in network.getUnconnectedOutLayers()]

# Минимальная вероятность для исключения слабых обнаружений
probability_minimum = 0.1

# Установка порога для фильтрации слабых ограничивающих рамок методом неперекрывающихся максимумов
threshold = 0.1

# Генерация цветов для ограничивающих рамок
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Чтение видео из файла с помощью объекта VideoCapture
# video = cv2.VideoCapture('input/car_data/70maiMiniDashCam-Dzien.mp4')
# video = cv2.VideoCapture('input/car_data/dusseldorf_test.mp4')
# video = cv2.VideoCapture('input/car_data/DODRX8W(lusterko)-roadtestwsonecznydzien_podsonce1080p30.mp4')
video = cv2.VideoCapture('input/traffic-signs-dataset-in-yolo-format/traffic-sign-to-test.mp4')

writer = None

# Размеры кадра
height, width = None, None

# Установка размера по умолчанию для графиков
plt.rcParams['figure.figsize'] = (3, 3)

total_frames = 0
total_time = 0

while True:
    # Захват кадров по одному
    ret, frame = video.read()

    # Если кадр не был получен
    if not ret:
        break

    # Получение размеров кадра
    if width is None or height is None:
        height, width = frame.shape[:2]

    # Blob из текущего кадра
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    total_frames += 1
    total_time += end - start

    print('Кадр {0} занял {1:.5f} секунд'.format(total_frames, end - start))

    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Проход по всем выходным слоям
    for result in output_from_network:
        # Проход по всем обнаружениям из текущего выходного слоя
        for detected_objects in result:
            # Получение вероятностей классов для текущего обнаруженного объекта
            scores = detected_objects[5:]
            # Получение индекса класса с максимальным значением вероятности
            class_current = np.argmax(scores)
            # Получение значения вероятности для определенного класса
            confidence_current = scores[class_current]
            # Исключение слабых предсказаний по минимальной вероятности
            if confidence_current > probability_minimum:
                try:
                    # Масштабирование координат ограничивающей рамки до исходного размера кадра
                    box_current = detected_objects[0:4] * np.array([width, height, width, height])

                    # Получение координат верхнего левого угла
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)
                except Exception as e:
                    print(e)

    # Реализация неперекрывающегося максимума заданных ограничивающих рамок
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    # Проверка, остался ли какой-то обнаруженный объект
    if len(results) > 0:
        # Проход по индексам результатов
        for i in results.flatten():
            # Координаты ограничивающей рамки, ее ширина и высота
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Вырезаем фрагмент с дорожным знаком
            c_ts = frame[y_min:y_min + int(box_height), x_min:x_min + int(box_width), :]

            if c_ts.shape[:1] == (0,) or c_ts.shape[1:2] == (0,):
                pass
            else:
                # Получение предобработанного blob с дорожным знаком нужной формы
                blob_ts = cv2.dnn.blobFromImage(c_ts, 1 / 255.0, size=(32, 32), swapRB=True, crop=False)
                blob_ts[0] = blob_ts[0, :, :, :] - mean['mean_image_rgb']
                blob_ts = blob_ts.transpose(0, 2, 3, 1)

                # Подача в модель Keras CNN для получения предсказанной метки среди 43 классов
                scores = model.predict(blob_ts)

                # Оценки даны для изображения с 43 числами предсказаний для каждого класса
                # Получение только одного класса с максимальным значением
                prediction = np.argmax(scores)

                # Цвет для текущей ограничивающей рамки
                colour_box_current = colors[class_numbers[i]].tolist()

                # Рисование ограничивающей рамки на исходном текущем кадре
                cv2.rectangle(frame, (x_min, y_min),
                              (x_min + box_width, y_min + box_height),
                              colour_box_current, 2)

                # Подготовка текста с меткой и вероятностью для текущей ограничивающей рамки
                text_box_current = '{}: {:.4f}'.format(labels['SignName'][prediction],
                                                       confidences[i])

                # Добавление текста с меткой и уверенностью на исходное изображение
                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('result__0.mp4', fourcc, 25,
                                 (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)

video.release()
writer.release()

print('Общее время обработки видеофайла {0:.5f} секунд'.format(total_time))
print('Общее количество кадров', total_frames)
