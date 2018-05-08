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
import pandas as pd

AVG_D1_Anger = 0.12391364155169986
Max_D1_Anger = 0.2572634421365949
Min_D1_Anger = 0.0
STD_D1_Anger = 0.035604259471758985
AVG_D2_Anger = 0.4696075976526778
Max_D2_Anger = 0.6683467291250962
Min_D2_Anger = 0.32483335803992286
STD_D2_Anger = 0.035604259471758985
AVG_D3_Anger = 0.8208555275812098
Max_D3_Anger = 1.2373154051984874
Min_D3_Anger = 0.4637421000881422
STD_D3_Anger = 0.035604259471758985
AVG_D4_Anger = 0.16033937556541833
Max_D4_Anger = 0.9279803573221784
Min_D4_Anger = 0.0
STD_D4_Anger = 0.035604259471758985
AVG_D5_Anger = 1.1466218540322337
Max_D5_Anger = 1.9298371421686333
Min_D5_Anger = 0.7746278291762028
STD_D5_Anger = 0.035604259471758985

AVG_D1_Disgust = 0.11190812756667812
Max_D1_Disgust = 0.23399066719890002
Min_D1_Disgust = 0.014512677579017377
STD_D1_Disgust = 0.040318611612492145
AVG_D2_Disgust = 0.46827281951159694
Max_D2_Disgust = 0.5833287003164145
Min_D2_Disgust = 0.3925823912839405
STD_D2_Disgust = 0.040318611612492145
AVG_D3_Disgust = 0.8192606070708873
Max_D3_Disgust = 1.1764705882352942
Min_D3_Disgust = 0.5784228974593904
STD_D3_Disgust = 0.040318611612492145
AVG_D4_Disgust = 0.11719313435554182
Max_D4_Disgust = 0.4441020371191435
Min_D4_Disgust = 0.0
STD_D4_Disgust = 0.040318611612492145
AVG_D5_Disgust = 1.1202426111103982
Max_D5_Disgust = 1.7998922276527947
Min_D5_Disgust = 0.8627137302342943
STD_D5_Disgust = 0.040318611612492145

AVG_D1_Fear = 0.13701074204244504
Max_D1_Fear = 0.2708484285643664
Min_D1_Fear = 0.014512677579017377
STD_D1_Fear = 0.04154064336614447
AVG_D2_Fear = 0.47984082001131434
Max_D2_Fear = 0.6226109722866823
Min_D2_Fear = 0.3525653824049223
STD_D2_Fear = 0.04154064336614447
AVG_D3_Fear = 0.8248149910169387
Max_D3_Fear = 1.2903225806451613
Min_D3_Fear = 0.4629899486900899
STD_D3_Fear = 0.04154064336614447
AVG_D4_Fear = 0.14681604601520945
Max_D4_Fear = 0.8483899914462799
Min_D4_Fear = 0.0
STD_D4_Fear = 0.04154064336614447
AVG_D5_Fear = 1.1331166693119874
Max_D5_Fear = 1.755796624200967
Min_D5_Fear = 0.8013390599347519
STD_D5_Fear = 0.04154064336614447

AVG_D1_Joy = 0.11466090482546239
Max_D1_Joy = 0.24820974816670166
Min_D1_Joy = 0.0
STD_D1_Joy = 0.033463324240914714
AVG_D2_Joy = 0.49561225672332654
Max_D2_Joy = 0.6787534453324541
Min_D2_Joy = 0.31906064576588006
STD_D2_Joy = 0.033463324240914714
AVG_D3_Joy = 0.9999279705170847
Max_D3_Joy = 1.5505272459038812
Min_D3_Joy = 0.4624349002139087
STD_D3_Joy = 0.033463324240914714
AVG_D4_Joy = 0.18566383389272834
Max_D4_Joy = 0.9210362180002667
Min_D4_Joy = 0.0
STD_D4_Joy = 0.033463324240914714
AVG_D5_Joy = 0.9918034963613195
Max_D5_Joy = 2.0689655172413794
Min_D5_Joy = 0.6678592727153194
STD_D5_Joy = 0.033463324240914714

AVG_D1_Sadness = 0.12078021641780236
Max_D1_Sadness = 0.2491364395612199
Min_D1_Sadness = 0.016638965801613444
STD_D1_Sadness = 0.03604623346862226
AVG_D2_Sadness = 0.47271444754198455
Max_D2_Sadness = 0.604845499011103
Min_D2_Sadness = 0.335605062279627
STD_D2_Sadness = 0.03604623346862226
AVG_D3_Sadness = 0.8085215401728708
Max_D3_Sadness = 1.2352117691761126
Min_D3_Sadness = 0.4415897845780168
STD_D3_Sadness = 0.03604623346862226
AVG_D4_Sadness = 0.08412142553440585
Max_D4_Sadness = 0.7019863897427859
Min_D4_Sadness = 0.0
STD_D4_Sadness = 0.03604623346862226
AVG_D5_Sadness = 1.1298852170864124
Max_D5_Sadness = 1.7710375531239897
Min_D5_Sadness = 0.7520088830047584
STD_D5_Sadness = 0.03604623346862226


AVG_D1_Surprise = 0.17224823229171185
Max_D1_Surprise = 0.26751282908862994
Min_D1_Surprise = 0.0547996624351191
STD_D1_Surprise = 0.03302752689237317

AVG_D2_Surprise = 0.4903754021209849
Max_D2_Surprise = 0.6020928060149454
Min_D2_Surprise = 0.34876544547944754
STD_D2_Surprise = 0.03302752689237317

AVG_D3_Surprise = 0.7585095565719763
Max_D3_Surprise = 1.278036689729597
Min_D3_Surprise = 0.45569620253164556
STD_D3_Surprise = 0.03302752689237317

AVG_D4_Surprise = 0.21308887635440873
Max_D4_Surprise = 0.8862089329795296
Min_D4_Surprise = 0.0
STD_D4_Surprise = 0.03302752689237317

AVG_D5_Surprise = 1.1718834967126281
Max_D5_Surprise = 1.8147739222897195
Min_D5_Surprise = 0.7766782052890988
STD_D5_Surprise = 0.03302752689237317


AVG_D1_Neutral = 0.132009049301306
Max_D1_Neutral = 0.2616841974223755
Min_D1_Neutral = 0.013099083795137854
STD_D1_Neutral = 0.03185885525776292
AVG_D2_Neutral = 0.4816606968267937
Max_D2_Neutral = 0.6298723622291464
Min_D2_Neutral = 0.33917080346043504
STD_D2_Neutral = 0.03185885525776292
AVG_D3_Neutral = 0.8006093687005107
Max_D3_Neutral = 1.1863425914930998
Min_D3_Neutral = 0.5031486596676551
STD_D3_Neutral = 0.03185885525776292
AVG_D4_Neutral = 0.05847609372132165
Max_D4_Neutral = 0.6382524546263842
Min_D4_Neutral = 0.0
STD_D4_Neutral = 0.03185885525776292
AVG_D5_Neutral = 1.1191068394133334
Max_D5_Neutral = 2.052038174762215
Min_D5_Neutral = 0.7756185729167818
STD_D5_Neutral = 0.03185885525776292


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
    #training_set = [row[1:-1] for row in rows if row[-1] == 'Training']
    #training_emotions = [int(row[0]) for row in rows if row[-1] == 'Training']

    data = pd.read_csv(dataset_file).values
    training_emotions = [data[i,0] for i in range(len(data)) if data[i,2] == 'Training']
    training_set = [np.fromstring(data[i,1], dtype=np.uint8, sep=' ') for i in range(len(data)) if data[i,2] == 'Training']
    test_emotions = [data[i,0] for i in range(len(data)) if data[i,2] == 'PublicTest' or data[i,2] == 'PrivateTest']
    test_set = [np.fromstring(data[i,1], dtype=np.uint8, sep=' ') for i in range(len(data)) if data[i,2] == 'PublicTest' or data[i,2] == 'PrivateTest']

    # reshaping into 48x48x1 image
    training_images = [i.reshape((48, 48)) for i in training_set]
    test_images = [i.reshape((48, 48)) for i in test_set]

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    # see https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_file)

    # for emot in range(0,7):
    #     print("\n")
    #     print('[', emot, ']')
    #     #initialize variables to get the average/min/max characterictic distances for expression
    #     D1_chr = []
    #     D2_chr = []
    #     D3_chr = []
    #     D4_chr = []
    #     D5_chr = []
    
    guesses = 0
    correct_guesses = 0
    the_list = []
    max_accuracy = -1
    max_a = -1
    max_b = -1
    max_c = -1
    max_d = -1
    max_e = -1
    for a in range(1,10):
        for b in range(1,10):
            for c in range(1,10):
                for d in range(1,10):
                    for e in range(1,10):
                        if a + b + c + d + e != 10:
                            continue
                        print(a,b,c,d,e, sep = '\t')
                        for (i, image) in enumerate(test_images):
                            #print(i, "\t", len(training_images))
                            #print(i, "\t", len(test_images))
                            # if training_emotions[i] != 6:
                            #         continue

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
                            # for (x, y) in landmarks:
                            #     cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

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

                            confidence = []

                            influence = True
                            if influence is True:
                                aa = .2
                                bb = .1
                                cc = .3
                                dd = .2
                                ee = .2
                            else:
                                aa = 1
                                bb = 1
                                cc = 1
                                dd = 1
                                ee = 1

                            C1 = 1 - abs(AVG_D1_Anger - D1)
                            C2 = 1 - abs(AVG_D2_Anger - D2)
                            C3 = 1 - abs(AVG_D3_Anger - D3)
                            C4 = 1 - abs(AVG_D4_Anger - D4)
                            C5 = 1 - abs(AVG_D5_Anger - D5)
                            confidence.append(sum([aa * C1, bb * C2, cc * C3, dd * C4, ee * C5]))

                            C1 = 1 - abs(AVG_D1_Disgust - D1)
                            C2 = 1 - abs(AVG_D2_Disgust - D2)
                            C3 = 1 - abs(AVG_D3_Disgust - D3)
                            C4 = 1 - abs(AVG_D4_Disgust - D4)
                            C5 = 1 - abs(AVG_D5_Disgust - D5)
                            confidence.append(sum([aa * C1, bb * C2, cc * C3, dd * C4, ee * C5]))

                            C1 = 1 - abs(AVG_D1_Fear - D1)
                            C2 = 1 - abs(AVG_D2_Fear - D2)
                            C3 = 1 - abs(AVG_D3_Fear - D3)
                            C4 = 1 - abs(AVG_D4_Fear - D4)
                            C5 = 1 - abs(AVG_D5_Fear - D5)
                            confidence.append(sum([aa * C1, bb * C2, cc * C3, dd * C4, ee * C5]))

                            C1 = 1 - abs(AVG_D1_Joy - D1)
                            C2 = 1 - abs(AVG_D2_Joy - D2)
                            C3 = 1 - abs(AVG_D3_Joy - D3)
                            C4 = 1 - abs(AVG_D4_Joy - D4)
                            C5 = 1 - abs(AVG_D5_Joy - D5)
                            confidence.append(sum([aa * C1, bb * C2, cc * C3, dd * C4, ee * C5]))

                            C1 = 1 - abs(AVG_D1_Sadness - D1)
                            C2 = 1 - abs(AVG_D2_Sadness - D2)
                            C3 = 1 - abs(AVG_D3_Sadness - D3)
                            C4 = 1 - abs(AVG_D4_Sadness - D4)
                            C5 = 1 - abs(AVG_D5_Sadness - D5)
                            confidence.append(sum([aa * C1, bb * C2, cc * C3, dd * C4, ee * C5]))
                            
                            C1 = 1 - abs(AVG_D1_Surprise - D1)
                            C2 = 1 - abs(AVG_D2_Surprise - D2)
                            C3 = 1 - abs(AVG_D3_Surprise - D3)
                            C4 = 1 - abs(AVG_D4_Surprise - D4)
                            C5 = 1 - abs(AVG_D5_Surprise - D5)
                            confidence.append(sum([aa * C1, bb * C2, cc * C3, dd * C4, ee * C5]))

                            C1 = 1 - abs(AVG_D1_Neutral - D1)
                            C2 = 1 - abs(AVG_D2_Neutral - D2)
                            C3 = 1 - abs(AVG_D3_Neutral - D3)
                            C4 = 1 - abs(AVG_D4_Neutral - D4)
                            C5 = 1 - abs(AVG_D5_Neutral - D5)
                            confidence.append(sum([aa * C1, bb * C2, cc * C3, dd * C4, ee * C5]))

                            # calculate distance from average neutral expressions
                            L1 = D1 - AVG_D1_Neutral
                            L2 = D2 - AVG_D2_Neutral
                            L3 = D3 - AVG_D3_Neutral
                            L4 = D4 - AVG_D4_Neutral
                            L5 = D5 - AVG_D5_Neutral

                            # the_list.append(L1)
                            # the_list.append(L2)
                            # the_list.append(L3)
                            # the_list.append(L4)
                            # the_list.append(L5)
                            

                            state = [0,0,0,0,0]
                            p = 0.11730329293977496
                            p1 = 0.039523999212758336
                            p2 = 0.04080891538920051
                            p3 = 0.14421632072267224
                            p4 = 0.12908862310308516
                            p5 = 0.14223306716801098 

                            #[2,2,-2,2,2]
                            #[1, 0, -1, 2, 0]
                            
                            if  L1 < -3 * p1:
                                state[0] = -2
                            elif L1 < -1 * p1:
                                state[0] = -1
                            elif L1 < 1 * p1:
                                state[0] = 0
                            elif L1 < 3 * p1:
                                state[0] = 1
                            else:
                                state[0] = 2

                            if  L2 < -3 * p2:
                                state[1] = -2
                            elif L2 < -1 * p2:
                                state[1] = -1
                            elif L2 < 1 * p2:
                                state[1] = 0
                            elif L2 < 3 * p2:
                                state[1] = 1
                            else:
                                state[1] = 2

                            if  L3 < -3 * p3:
                                state[2] = -2
                            elif L3 < -1 * p3:
                                state[2] = -1
                            elif L3 < 1 * p3:
                                state[2] = 0
                            elif L3 < 3 * p3:
                                state[2] = 1
                            else:
                                state[2] = 2

                            if  L4 < -3 * p4:
                                state[3] = -2
                            elif L4 < -1 * p4:
                                state[3] = -1
                            elif L4 < 1 * p4:
                                state[3] = 0
                            elif L4 < 3 * p4:
                                state[3] = 1
                            else:
                                state[3] = 2

                            if  L5 < -3 * p5:
                                state[4] = -2
                            elif L5 < -1 * p5:
                                state[4] = -1
                            elif L5 < 1 * p5:
                                state[4] = 0
                            elif L5 < 3 * p5:
                                state[4] = 1
                            else:
                                state[4] = 2


                            guesses += 1
                            # [0"Anger", 1"Disgust", 2"Fear", 3"Joy", 
                            #  4"Sadness", 5"Surprise", 6"Neutral"]
                            emotion_guess = -1
                            if state == [-2,-1,2,2,-2]:
                                emotion_guess = 3
                            elif state == [2,2,-2,2,2]:
                                emotion_guess = 5
                            elif state == [-2,-2,1,2,-1]:
                                emotion_guess = 1
                            elif state == [2,-2,0,-1,0]:
                                emotion_guess = 0
                            elif state == [-2,2,0,2,0]:
                                emotion_guess = 4
                            elif state == [1,1,-1,1,0]:
                                emotion_guess = 2
                            elif state == [0,0,0,0,0]:
                                emotion_guess = 6
                            else:
                                emotion_guess = np.argsort(confidence)[-1]
                            # Check if accurate guess
                            if test_emotions[i] == emotion_guess:
                                correct_guesses += 1

                            # # [0"Anger", 1"Disgust", 2"Fear", 3"Joy", 
                            # # #  4"Sadness", 5"Surprise", 6"Neutral"]
                            # print("Actual emotion:", emotions[test_emotions[i]])
                            # print("Guess:", emotions[emotion_guess])
                            # print("top two:", emotions[np.argsort(confidence)[-1]], emotions[np.argsort(confidence)[-2]])
                            # print("Anger:", confidence[0]/5*100)
                            # print("Disgust:", confidence[1]/5*100)
                            # print("Fear:", confidence[2]/5*100)
                            # print("Joy:", confidence[3]/5*100)
                            # print("Sadness:", confidence[4]/5*100)
                            # print("Surprise:", confidence[5]/5*100)
                            # print("Neutral:", confidence[6]/5*100)
                            # print([L1, L2, L3, L4, L5])
                            # print(state)
                            # #show the output image with the face detections + facial landmarks
                            # cv2.imshow("Output", image)

                            # if cv2.waitKey(0) == ord('q'):
                            #     quit()
                            # else:
                            #     continue
                        accuracy = 0
                        accuracy = (correct_guesses / guesses) * 100
                        correct_guesses = 0
                        guesses = 0
                        if accuracy >= max_accuracy:
                            max_accuracy = accuracy
                            max_a = a/10
                            max_b = b/10
                            max_c = c/10
                            max_d = d/10
                            max_e = e/10
                        print("Current:", accuracy)
                        print("max_accuracy", max_accuracy)
                        print("Maxes: ", max_a, max_b, max_c, max_d, max_e, sep='\t')
                        #print(guesses, "::", len(test_images))
                        #print("Training Accuracy:", accuracy)
                        #print("STD_L =", statistics.stdev(the_list))

    

                            # #Add charateristic distances of emotions to list
                            # if training_emotions[i] == emot:
                            #     D1_chr.append(D1)
                            #     D2_chr.append(D2)
                            #     D3_chr.append(D3)
                            #     D4_chr.append(D4)
                            #     D5_chr.append(D5)

                            

                            # #Calculate average/max/min neutral L distance emotion chracteristics
                            # avg_D1_chr = sum(D1_chr) / len(D1_chr)
                            # avg_D2_chr = sum(D2_chr) / len(D2_chr)
                            # avg_D3_chr = sum(D3_chr) / len(D3_chr)
                            # avg_D4_chr = sum(D4_chr) / len(D4_chr)
                            # avg_D5_chr = sum(D5_chr) / len(D5_chr)
                            
                            # # print("AVG_D1_" + emotions[emot] + " =", avg_D1_chr)
                            # # print("Max_D1_" + emotions[emot] + " =", max(D1_chr))
                            # # print("Min_D1_" + emotions[emot] + " =", min(D1_chr))
                            # # print("STD_D1_" + emotions[emot] + " =", statistics.stdev(D1_chr))

                            # # print("AVG_D2_" + emotions[emot] + " =", avg_D2_chr)
                            # # print("Max_D2_" + emotions[emot] + " =", max(D2_chr))
                            # # print("Min_D2_" + emotions[emot] + " =", min(D2_chr))
                            # # print("STD_D2_" + emotions[emot] + " =", statistics.stdev(D1_chr))

                            # # print("AVG_D3_" + emotions[emot] + " =", avg_D3_chr)
                            # # print("Max_D3_" + emotions[emot] + " =", max(D3_chr))
                            # # print("Min_D3_" + emotions[emot] + " =", min(D3_chr))
                            # # print("STD_D3_" + emotions[emot] + " =", statistics.stdev(D1_chr))

                            # # print("AVG_D4_" + emotions[emot] + " =", avg_D4_chr)
                            # # print("Max_D4_" + emotions[emot] + " =", max(D4_chr))
                            # # print("Min_D4_" + emotions[emot] + " =", min(D4_chr))
                            # # print("STD_D4_" + emotions[emot] + " =", statistics.stdev(D1_chr))

                            # # print("AVG_D5_" + emotions[emot] + " =", avg_D5_chr)
                            # # print("Max_D5_" + emotions[emot] + " =", max(D5_chr))
                            # # print("Min_D5_" + emotions[emot] + " =", min(D5_chr))
                            # # print("STD_D5_" + emotions[emot] + " =", statistics.stdev(D1_chr))
                            
                            # print("AVG_D1_" + emotions[emot] + " =", avg_D1_chr)
                            # print("Max_D1_" + emotions[emot] + " =", max(D1_chr))
                            # print("Min_D1_" + emotions[emot] + " =", min(D1_chr))
                            # print("STD_D1_" + emotions[emot] + " =", statistics.stdev(D1_chr))

                            # print("AVG_D2_" + emotions[emot] + " =", avg_D2_chr)
                            # print("Max_D2_" + emotions[emot] + " =", max(D2_chr))
                            # print("Min_D2_" + emotions[emot] + " =", min(D2_chr))
                            # print("STD_D2_" + emotions[emot] + " =", statistics.stdev(D1_chr))

                            # print("AVG_D3_" + emotions[emot] + " =", avg_D3_chr)
                            # print("Max_D3_" + emotions[emot] + " =", max(D3_chr))
                            # print("Min_D3_" + emotions[emot] + " =", min(D3_chr))
                            # print("STD_D3_" + emotions[emot] + " =", statistics.stdev(D1_chr))

                            # print("AVG_D4_" + emotions[emot] + " =", avg_D4_chr)
                            # print("Max_D4_" + emotions[emot] + " =", max(D4_chr))
                            # print("Min_D4_" + emotions[emot] + " =", min(D4_chr))
                            # print("STD_D4_" + emotions[emot] + " =", statistics.stdev(D1_chr))

                            # print("AVG_D5_" + emotions[emot] + " =", avg_D5_chr)
                            # print("Max_D5_" + emotions[emot] + " =", max(D5_chr))
                            # print("Min_D5_" + emotions[emot] + " =", min(D5_chr))
                            # print("STD_D5_" + emotions[emot] + " =", statistics.stdev(D1_chr))
                        



if __name__ == "__main__":
    main()



