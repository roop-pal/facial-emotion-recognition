# import the necessary packages
from imutils import face_utils
import csv
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
import imutils
import dlib
import cv2
import statistics

AVG_D1_NEUTRAL = 0.132009049301306
AVG_D2_NEUTRAL = 0.4816606968267937
AVG_D3_NEUTRAL = 0.8006093687005107
AVG_D4_NEUTRAL = 0.28650983575677974
AVG_D5_NEUTRAL = 1.1191068394133334

AVG_L1_Angry = -0.008095407749606239
Max_L1_Angry = 0.1252543928352889
Min_L1_Angry = -0.132009049301306
STD_L1_Angry = 0.035604259471758985
AVG_L2_Angry = -0.01205309917411583
Max_L2_Angry = 0.1866860322983025
Min_L2_Angry = -0.15682733878687083
STD_L2_Angry = 0.035604259471758985
AVG_L3_Angry = 0.020246158880701356
Max_L3_Angry = 0.4367060364979767
Min_L3_Angry = -0.3368672686123685
STD_L3_Angry = 0.035604259471758985
AVG_L4_Angry = 0.08982321953498053
Max_L4_Angry = 0.8630478022828736
Min_L4_Angry = -0.28650983575677974
STD_L4_Angry = 0.035604259471758985
AVG_L5_Angry = 0.027515014618902274
Max_L5_Angry = 0.8107303027552999
Min_L5_Angry = -0.3444790102371307
STD_L5_Angry = 0.035604259471758985


AVG_L1_Disgust = -0.0201009217346278
Max_L1_Disgust = 0.10198161789759402
Min_L1_Disgust = -0.11749637172228862
STD_L1_Disgust = 0.040318611612492145
AVG_L2_Disgust = -0.013387877315196762
Max_L2_Disgust = 0.1016680034896208
Min_L2_Disgust = -0.08907830554285318
STD_L2_Disgust = 0.040318611612492145
AVG_L3_Disgust = 0.018651238370376334
Max_L3_Disgust = 0.37586121953478346
Min_L3_Disgust = -0.22218647124112034
STD_L3_Disgust = 0.040318611612492145
AVG_L4_Disgust = 0.04765446208671295
Max_L4_Disgust = 0.4362219529703388
Min_L4_Disgust = -0.2542517712406507
STD_L4_Disgust = 0.040318611612492145
AVG_L5_Disgust = 0.0011357716970645398
Max_L5_Disgust = 0.6807853882394612
Min_L5_Disgust = -0.2563931091790391
STD_L5_Disgust = 0.040318611612492145


AVG_L1_Fear = 0.005001692741138909
Max_L1_Fear = 0.13883937926306042
Min_L1_Fear = -0.11749637172228862
STD_L1_Fear = 0.04154064336614447
AVG_L2_Fear = -0.001819876815480081
Max_L2_Fear = 0.14095027545988859
Min_L2_Fear = -0.1290953144218714
STD_L2_Fear = 0.04154064336614447
AVG_L3_Fear = 0.024205622316427217
Max_L3_Fear = 0.48971321194465056
Min_L3_Fear = -0.3376194200104208
STD_L3_Fear = 0.04154064336614447
AVG_L4_Fear = 0.07708206602781227
Max_L4_Fear = 0.9190653677849179
Min_L4_Fear = -0.28650983575677974
STD_L4_Fear = 0.04154064336614447
AVG_L5_Fear = 0.014009829898652302
Max_L5_Fear = 0.6366897847876336
Min_L5_Fear = -0.3177677794785816
STD_L5_Fear = 0.04154064336614447


AVG_L1_Happy = -0.017348144475843943
Max_L1_Happy = 0.11620069886539566
Min_L1_Happy = -0.132009049301306
STD_L1_Happy = 0.033463324240914714
AVG_L2_Happy = 0.013951559896533313
Max_L2_Happy = 0.1970927485056604
Min_L2_Happy = -0.16260005106091363
STD_L2_Happy = 0.033463324240914714
AVG_L3_Happy = 0.1993186018165767
Max_L3_Happy = 0.7499178772033706
Min_L3_Happy = -0.338174468486602
STD_L3_Happy = 0.033463324240914714
AVG_L4_Happy = 0.11306738263516661
Max_L4_Happy = 0.9043600428594873
Min_L4_Happy = -0.2672932852035085
STD_L4_Happy = 0.033463324240914714
AVG_L5_Happy = -0.12730334305201182
Max_L5_Happy = 0.949858677828046
Min_L5_Happy = -0.451247566698014
STD_L5_Happy = 0.033463324240914714


AVG_L1_Sad = -0.011228832883503128
Max_L1_Sad = 0.1171273902599139
Min_L1_Sad = -0.11537008349969255
STD_L1_Sad = 0.03604623346862226
AVG_L2_Sad = -0.008946249284809178
Max_L2_Sad = 0.12318480218430927
Min_L2_Sad = -0.1460556345471667
STD_L2_Sad = 0.03604623346862226
AVG_L3_Sad = 0.007912171472362728
Max_L3_Sad = 0.43460240047560195
Min_L3_Sad = -0.3590195841224939
STD_L3_Sad = 0.03604623346862226
AVG_L4_Sad = 0.010217975498046877
Max_L4_Sad = 0.6530721113155404
Min_L4_Sad = -0.28650983575677974
STD_L4_Sad = 0.03604623346862226
AVG_L5_Sad = 0.010778377673078125
Max_L5_Sad = 0.6519307137106563
Min_L5_Sad = -0.367097956408575
STD_L5_Sad = 0.03604623346862226


AVG_L1_Surprise = 0.0402391829904058
Max_L1_Surprise = 0.13550377978732395
Min_L1_Surprise = -0.07720938686618689
STD_L1_Surprise = 0.03302752689237317
AVG_L2_Surprise = 0.008714705294192092
Max_L2_Surprise = 0.12043210918815173
Min_L2_Surprise = -0.13289525134734614
STD_L2_Surprise = 0.03302752689237317
AVG_L3_Surprise = -0.042099812128535676
Max_L3_Surprise = 0.47742732102908625
Min_L3_Surprise = -0.34491316616886514
STD_L3_Surprise = 0.03302752689237317
AVG_L4_Surprise = 0.1602453460461907
Max_L4_Surprise = 0.8540048328937415
Min_L4_Surprise = -0.28650983575677974
STD_L4_Surprise = 0.03302752689237317
AVG_L5_Surprise = 0.05277665729929616
Max_L5_Surprise = 0.695667082876386
Min_L5_Surprise = -0.34242863412423463
STD_L5_Surprise = 0.03302752689237317


AVG_L1_Neutral = 1.3038474732970043e-16
Max_L1_Neutral = 0.12967514812106948
Min_L1_Neutral = -0.11890996550616814
STD_L1_Neutral = 0.03185885525776292
AVG_L2_Neutral = 1.3443495561969695e-15
Max_L2_Neutral = 0.14821166540235275
Min_L2_Neutral = -0.14248989336635864
STD_L2_Neutral = 0.03185885525776292
AVG_L3_Neutral = -3.5158555886098246e-15
Max_L3_Neutral = 0.38573322279258915
Min_L3_Neutral = -0.2974607090328556
STD_L3_Neutral = 0.03185885525776292
AVG_L4_Neutral = 1.0834904551594714e-16
Max_L4_Neutral = 0.6216693173999714
Min_L4_Neutral = -0.28650983575677974
STD_L4_Neutral = 0.03185885525776292
AVG_L5_Neutral = 5.855179432802513e-16
Max_L5_Neutral = 0.9329313353488815
Min_L5_Neutral = -0.3434882664965516
STD_L5_Neutral = 0.03185885525776292


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
    D4 = dist(landmarks[62], landmarks[66]) / N
    #print(temp1, D4)
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
    emotions = ["Anger", "Disgust", "Fear",
                "Joy", "Sadness", "Surprise", "Neutral"]

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

    # for emot in range(0,7):
        # print("\n")
        # initialize variables to get the average/min/max characterictic distances for expression
        # D1_chr = []
        # D2_chr = []
        # D3_chr = []
        # D4_chr = []
        # D5_chr = []

    for (i, image) in enumerate(training_images):
        #print(i, "\t", len(training_images))
        #if training_emotions[i] != emot:
        #    continue

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

        # calculate offset from neutral expression
        L1 = D1 - AVG_D1_NEUTRAL
        L2 = D2 - AVG_D2_NEUTRAL
        L3 = D3 - AVG_D3_NEUTRAL
        L4 = D4 - AVG_D4_NEUTRAL
        L5 = D5 - AVG_D5_NEUTRAL

        state = [0, 0, 0, 0, 0, 0, 0]

        if L1 <= -1 * STD_L1_Neutral:
            state[0] = -2
        elif L1 < STD_L1_Neutral:
            state[0] = -1
        elif L1 < STD_L1_Neutral * 2:
            state[0] = 0
        elif L1 < STD_L1_Neutral * 3:
            state[0] = 1
        else:
            state[0] = 2

        if L2 <= -4 * STD_L2_Neutral:
            state[1] = -2
        elif L2 < -3 * STD_L2_Neutral:
            state[1] = -1
        elif L2 < STD_L2_Neutral:
            state[1] = 0
        elif L2 < STD_L2_Neutral * 2:
            state[1] = 1
        else:
            state[1] = 2

        if L3 <= -2 * STD_L3_Neutral:
            state[2] = -2
        elif L3 < -1 * STD_L3_Neutral:
            state[2] = -1
        elif L3 < STD_L3_Neutral:
            state[2] = 0
        elif L3 < STD_L3_Neutral * 2:
            state[2] = 1
        else:
            state[2] = 2

        if L4 <= -2 * STD_L4_Neutral:
            state[3] = -2
        elif L4 < -1 * STD_L4_Neutral:
            state[3] = -1
        elif L4 < STD_L4_Neutral:
            state[3] = 0
        elif L4 < STD_L4_Neutral * 2:
            state[3] = 1
        else:
            state[3] = 2

        if L5 <= -2 * STD_L5_Neutral:
            state[4] = -2
        elif L5 < -1 * STD_L5_Neutral:
            state[4] = -1
        elif L5 < STD_L5_Neutral:
            state[4] = 0
        elif L5 < STD_L5_Neutral * 2:
            state[4] = 1
        else:
            state[4] = 2
        
        guess_list = []
        if state == [-2, -1, 2, 2, -1]:
            guess_list.append("Joy")
        elif state == [2, 2, -2, 2, 2]:
            guess_list.append("Surprise")
        elif state == [-2, -2, 1, 2, -1]:
            guess_list.append("Disgust")
        elif state == [2, -2, 0, -1, 0]:
            guess_list.append("Anger")
        elif state == [-2, 2, 0, 2, 0]:
            guess_list.append("Sadness")
        elif state == [1, 1, -1, 1, 0]:
            guess_list.append("Fear")
        elif state == [0, 0, 0, 0, 0]:
            guess_list.append("Neutral")
        else:
            guess_list.append("Unknown")


        # find associated emotion
        print(L1, L2, L3, L4, L5, sep='\n')
        print("Emotion:", emotions[training_emotions[i]])
        print(state)
        print("Guess: ",' '.join(guess_list), end = "\n\n")

        # Add charateristic distances of emotions to list
        # if training_emotions[i] == emot:
        #     D1_chr.append(L1)
        #     D2_chr.append(L2)
        #     D3_chr.append(L3)
        #     D4_chr.append(L4)
        #     D5_chr.append(L5)

        #print(L1, L2, L3, L4, L5)

        

        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", image)

        if cv2.waitKey(0) == ord('q'):
           quit()
        else:
           continue

    # Calculate average/max/min neutral L distance emotion chracteristics
    # avg_D1_chr = sum(D1_chr) / len(D1_chr)
    # avg_D2_chr = sum(D2_chr) / len(D2_chr)
    # avg_D3_chr = sum(D3_chr) / len(D3_chr)
    # avg_D4_chr = sum(D4_chr) / len(D4_chr)
    # avg_D5_chr = sum(D5_chr) / len(D5_chr)

    # print("AVG_L1_" + emotions[emot] + " =", avg_D1_chr)
    # print("Max_L1_" + emotions[emot] + " =", max(D1_chr))
    # print("Min_L1_" + emotions[emot] + " =", min(D1_chr))
    # print("STD_L1_" + emotions[emot] + " =", statistics.stdev(D1_chr))

    # print("AVG_L2_" + emotions[emot] + " =", avg_D2_chr)
    # print("Max_L2_" + emotions[emot] + " =", max(D2_chr))
    # print("Min_L2_" + emotions[emot] + " =", min(D2_chr))
    # print("STD_L2_" + emotions[emot] + " =", statistics.stdev(D1_chr))

    # print("AVG_L3_" + emotions[emot] + " =", avg_D3_chr)
    # print("Max_L3_" + emotions[emot] + " =", max(D3_chr))
    # print("Min_L3_" + emotions[emot] + " =", min(D3_chr))
    # print("STD_L3_" + emotions[emot] + " =", statistics.stdev(D1_chr))

    # print("AVG_L4_" + emotions[emot] + " =", avg_D4_chr)
    # print("Max_L4_" + emotions[emot] + " =", max(D4_chr))
    # print("Min_L4_" + emotions[emot] + " =", min(D4_chr))
    # print("STD_L4_" + emotions[emot] + " =", statistics.stdev(D1_chr))

    # print("AVG_L5_" + emotions[emot] + " =", avg_D5_chr)
    # print("Max_L5_" + emotions[emot] + " =", max(D5_chr))
    # print("Min_L5_" + emotions[emot] + " =", min(D5_chr))
    # print("STD_L5_" + emotions[emot] + " =", statistics.stdev(D1_chr))




if __name__ == "__main__":
    main()



