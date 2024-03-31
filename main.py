import numpy as np
import pandas as pd
import cv2
import time
import matplotlib.pyplot as plt
import pickle

from keras.models import load_model

# Reading csv file with labels' names
# Loading two columns [0, 1] into Pandas dataFrame
labels = pd.read_csv('input/traffic-signs-preprocessed/label_names.csv')

model = load_model('model-3x3.h5')

# Loading mean image to use for preprocessing further
# Opening file for reading in binary mode
with open('input/traffic-signs-preprocessed/mean_image_rgb.pickle', 'rb') as total_frames:
    mean = pickle.load(total_frames, encoding='latin1')  # dictionary type

print(mean['mean_image_rgb'].shape)  # (3, 32, 32)

model.summary()

# Trained weights can be found in the course mentioned above
path_to_cfg = 'input/traffic-signs-dataset-in-yolo-format/yolov3_ts_test.cfg'
path_to_weights = 'input/car_data/znaki_rtx_final.weights'

path_to_cfg_markings = 'input/car_data/markings_test.cfg'
path_to_weights_markings = 'input/car_data/poziome_rtx_final.weights'

# Loading trained YOLO v3 weights and cfg configuration file by 'dnn' library from OpenCV
network = cv2.dnn.readNetFromDarknet(path_to_cfg, path_to_weights)

# To use with GPU
network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
network.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)

# Getting names of all YOLO v3 layers
layers_all = network.getLayerNames()
layers_names_output = [layers_all[i - 1] for i in network.getUnconnectedOutLayers()]
print(layers_names_output)

# Minimum probability to eliminate weak detections
probability_minimum = 0.1

# Setting threshold to filtering weak bounding boxes by non-maximum suppression
threshold = 0.1

# Generating colours for bounding boxes
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
colours_markings = np.random.randint(0, 255, size=(1, 3), dtype='uint8')

# Reading video from a file by VideoCapture object
video = cv2.VideoCapture('input/car_data/70maiMiniDashCam-Dzien.mp4')
# video = cv2.VideoCapture('input/car_data/DODRX8W(lusterko)-roadtestwsonecznydzien_podsonce1080p30.mp4')
# video = cv2.VideoCapture('input/traffic-signs-dataset-in-yolo-format/traffic-sign-to-test.mp4')

# Writer that will be used to write processed frames
writer = None

# Variables for spatial dimensions of the frames
h, w = None, None

###############################################################################

# Setting default size of plots
plt.rcParams['figure.figsize'] = (3, 3)

# Variable for counting total amount of frames
total_frames = 0

# Variable for counting total processing time
t = 0

# Catching frames in the loop
while True:
    # Capturing frames one-by-one
    ret, frame = video.read()

    # If the frame was not retrieved
    if not ret:
        break

    # Getting spatial dimensions of the frame for the first time
    if w is None or h is None:
        # Slicing two elements from tuple
        h, w = frame.shape[:2]

    # Blob from current frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Forward pass with blob through output layers
    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Increasing counters
    total_frames += 1
    t += end - start

    # Spent time for current frame
    print('Frame number {0} took {1:.5f} seconds'.format(total_frames, end - start))

    # Lists for detected bounding boxes, confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]
            # Eliminating weak predictions by minimum probability
            if confidence_current > probability_minimum:
                try:
                    # Scaling bounding box coordinates to the initial frame size
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Getting top left corner coordinates
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)
                except Exception as e:
                    print(e)

    # Implementing non-maximum suppression of given bounding boxes
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    # Checking if there is any detected object been left
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Bounding box coordinates, its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Cut fragment with Traffic Sign
            c_ts = frame[y_min:y_min + int(box_height), x_min:x_min + int(box_width), :]

            if c_ts.shape[:1] == (0,) or c_ts.shape[1:2] == (0,):
                pass
            else:
                # Getting preprocessed blob with Traffic Sign of needed shape
                blob_ts = cv2.dnn.blobFromImage(c_ts, 1 / 255.0, size=(32, 32), swapRB=True, crop=False)
                blob_ts[0] = blob_ts[0, :, :, :] - mean['mean_image_rgb']
                blob_ts = blob_ts.transpose(0, 2, 3, 1)

                # Feeding to the Keras CNN model to get predicted label among 43 classes
                scores = model.predict(blob_ts)

                # Scores is given for image with 43 numbers of predictions for each class
                # Getting only one class with maximum value
                prediction = np.argmax(scores)

                # Colour for current bounding box
                colour_box_current = colours[class_numbers[i]].tolist()

                # Drawing bounding box on the original current frame
                cv2.rectangle(frame, (x_min, y_min),
                              (x_min + box_width, y_min + box_height),
                              colour_box_current, 2)

                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format(labels['SignName'][prediction],
                                                       confidences[i])

                # Putting text with label and confidence on the original image
                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    # Initializing writer only once
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Writing current processed frame into the video file
        writer = cv2.VideoWriter('result_0_2.mp4', fourcc, 25,
                                 (frame.shape[1], frame.shape[0]), True)

    # Write processed current frame to the file
    writer.write(frame)

# Releasing video reader and writer
video.release()
writer.release()
