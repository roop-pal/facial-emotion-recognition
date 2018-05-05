# import the necessary packages
from imutils import face_utils
import csv
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import imutils
import dlib
import cv2

AVG_D1_NEUTRAL = 0.132009049301306
AVG_D2_NEUTRAL = 0.4816606968267937
AVG_D3_NEUTRAL = 0.8006093687005107
AVG_D4_NEUTRAL = 0.28650983575677974
AVG_D5_NEUTRAL = 1.1191068394133334

def shape_to_np(shape, dtype="int"):
    """
    convert dlib shape to np array 
    initialize the list of (x, y)-coordinates
    """
    coords = np.zeros((68, 2), dtype=dtype)

    # convert the 68 facial landmarks to a 2-tuple
    # of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def dist(point1, point2):
    """
    return distance between two points
    """
    (x1, y1) = point1
    (x2, y2) = point2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)


def N_dist(landmarks):
    """
    calculate vertical distance between irises (N)
    this is used to normalize the characteristic facial distances
    """
    temp1 = dist(landmarks[37], landmarks[43])
    temp2 = dist(landmarks[38], landmarks[44])
    temp3 = dist(landmarks[41], landmarks[47])
    temp4 = dist(landmarks[40], landmarks[46])
    N = (temp1 + temp2 + temp3 + temp4) / 4
    #print(temp1, temp2, temp3, temp4, N)
    return N


def D1_dist(landmarks, N):
    """
    D1 Eye opening, distance between upper and lower eyelids.
    """
    temp1 = dist(landmarks[37], landmarks[41]) / N
    temp2 = dist(landmarks[38], landmarks[40]) / N
    temp3 = dist(landmarks[43], landmarks[47]) / N
    temp4 = dist(landmarks[44], landmarks[46]) / N
    D1 = (temp1 + temp2 + temp3 + temp4) / 4
    #print(temp1, temp2, temp3, temp4, D1)
    return D1


def D2_dist(landmarks, N):
    """
    D2 Distance between the interior corner of the eye and 
    the interior corner of the eyebrow.
    """
    temp1 = dist(landmarks[21], landmarks[22]) / N
    temp2 = dist(landmarks[39], landmarks[42]) / N
    D2 = (temp1 + temp2) / 2
    #print(temp1, temp2, D2)
    return D2


def D3_dist(landmarks, N):
    """
    D3 Mouth opening width, distance between 
    left and right mouth corners 
    """
    D3 = dist(landmarks[48], landmarks[54]) / N
    # print(D3)
    return D3


def D4_dist(landmarks, N):
    """
    D4 Mouth opening height, distance between upper and lower lips.
    """
    temp1 = dist(landmarks[50], landmarks[58]) / N
    temp2 = dist(landmarks[52], landmarks[56]) / N
    D4 = (temp1 + temp2) / 2
    #print(temp1, temp2, D4)
    return D4


def D5_dist(landmarks, N):
    """
    D5 Distance between a corner of the mouth 
    and the corresponding external eye corner.
    """
    temp1 = dist(landmarks[36], landmarks[48]) / N
    temp2 = dist(landmarks[45], landmarks[54]) / N
    D5 = (temp1 + temp2) / 2
    #print(temp1, temp2, D5)
    return D5


def main():
    # initialize constants
    shape_predictor_file = "shape_predictor_68_face_landmarks.dat"
    dataset_file = "../fer2013/fer2013.csv"
    emotions = ["Angry", "Disgust", "Fear",
                "Happy", "Sad", "Surprise", "Neutral"]

    # open dataset
    csvr = csv.reader(open(dataset_file))

    header = next(csvr)

    rows = [list(row) for row in csvr]

    # get training images and emotions
    training_set = [row[1:-1] for row in rows if row[-1] == 'Training']
    training_emotions = [int(row[0]) for row in rows if row[-1] == 'Training']

    # transform training_set into 48 x 48 images
    training_images = []

    for i in training_set:
        # print (emotions[int(i[0])])
        tmp_image = np.fromstring(
            i[0], dtype=np.uint8, sep=" ").reshape((48, 48))

        training_images.append(tmp_image)

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    # see https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_file)

    # initialize variables to get the average characterictic distances for neutral expression
    D1_neutral = []
    D2_neutral = []
    D3_neutral = []
    D4_neutral = []
    D5_neutral = []

    for (i, image) in enumerate(training_images):
        print(i, "\t", len(training_images))
        # if training_emotions[i] != 6:  # neutral
        #     continue

        # detect faces in the grayscale image
        #image = imutils.resize(image, width=500)
        faces = detector(image, 1)

        if len(faces) == 0:
            #print("No faces found")
            continue
        elif len(faces) > 1:
            #print("Multiple faces found")
            continue
        else:
            faces = faces[0]

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        landmarks = predictor(image, faces)
        landmarks = shape_to_np(landmarks)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        #(x, y, w, h) = face_utils.rect_to_bb(face)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in landmarks:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        # calculate vertical distance between irises (N)
        # this is used to normalize the characteristic facial distances
        N = N_dist(landmarks)

        # Calculate the following:

        # D1 Eye opening, distance between upper and lower eyelids.
        D1 = D1_dist(landmarks, N)

        # D2 Distance between the interior corner of the eye and
        # the interior corner of the eyebrow.
        D2 = D2_dist(landmarks, N)

        # D3 Mouth opening width, distance between left and right mouth corners
        D3 = D3_dist(landmarks, N)

        # D4 Mouth opening height, distance between upper and lower lips.
        D4 = D4_dist(landmarks, N)

        # D5 Distance between a corner of the mouth
        # and the corresponding external eye corner.
        D5 = D5_dist(landmarks, N)

        # Add charateristic distances of neutral emotions to list
        # if training_emotions[i] == 6:  # neutral
        #     D1_neutral.append(D1)
        #     D2_neutral.append(D2)
        #     D3_neutral.append(D3)
        #     D4_neutral.append(D4)
        #     D5_neutral.append(D5)

        # calculate offset from neutral expression
        L1 = D1 - AVG_D1_NEUTRAL
        L2 = D2 - AVG_D2_NEUTRAL
        L3 = D3 - AVG_D3_NEUTRAL
        L4 = D4 - AVG_D4_NEUTRAL
        L5 = D5 - AVG_D5_NEUTRAL

        print(L1, L2, L3, L4, L5)

        # find associated emotion
        print("Emotion:", emotions[training_emotions[i]])

        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", image)

        if cv2.waitKey(0) == ord('q'):
           quit()
        else:
           continue

    # Calculate average neutral emotion chracteristics
    # avg_D1_neutral = sum(D1_neutral) / len(D1_neutral)
    # avg_D2_neutral = sum(D2_neutral) / len(D2_neutral)
    # avg_D3_neutral = sum(D3_neutral) / len(D3_neutral)
    # avg_D4_neutral = sum(D4_neutral) / len(D4_neutral)
    # avg_D5_neutral = sum(D5_neutral) / len(D5_neutral)
    # print("Avg D1 Neutral:", avg_D1_neutral)
    # print("Avg D2 Neutral:", avg_D2_neutral)
    # print("Avg D3 Neutral:", avg_D3_neutral)
    # print("Avg D4 Neutral:", avg_D4_neutral)
    # print("Avg D5 Neutral:", avg_D5_neutral)


if __name__ == "__main__":
    main()



