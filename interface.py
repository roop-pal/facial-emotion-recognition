from imutils.video import VideoStream
from imutils import face_utils
import datetime
import imutils
import time
import dlib
import cv2
import numpy as np
import statistics
import sys
sys.path.insert(0, "./deterministic")
from measured_values import *
sys.path.insert(0, "./unsupervised")
sys.path.insert(0, "./supervised")
from main import guess
from unsupervised_classify import UnsupervisedClassification
from math import sqrt
import tensorflow as tf
import csv

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

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def video_demo():
	# Load facial predictor
	print("[INFO] loading facial landmark predictor...")
	shape_predictor_file = "shape_predictor_68_face_landmarks.dat"
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor_file)

	# open camera
	print("[INFO] opening webcam...")
	vs = VideoStream(0).start()
	time.sleep(1)
	img_counter = 0

	emotions = ["Anger", "Disgust", "Fear",
					"Joy", "Sadness", "Surprise", "Neutral"]

	while True:
		# get frame, resize it, and convert it to gray
		frame = vs.read()
		frame = imutils.resize(frame, width=800)

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		faces = detector(gray, 0)

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
		landmarks = predictor(frame, faces)
		landmarks = shape_to_np(landmarks)

		#loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in landmarks:
			cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

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

		# Calculate confidence scores of each emotion
		confidence = []

		# Certain characterictic distances seem to matter more,
		# therefore give them different weights if influence = True
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

		# Calculate confidence percentage based on how far each
		# characteristic distance is from the calculate average distances
		# for each emotion
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

		# Calculate distance from average neutral expressions
		L1 = D1 - AVG_D1_Neutral
		L2 = D2 - AVG_D2_Neutral
		L3 = D3 - AVG_D3_Neutral
		L4 = D4 - AVG_D4_Neutral
		L5 = D5 - AVG_D5_Neutral

		# Determine a "state" for L distances based on calculated
		# partions from -1 to 1
		state = [0,0,0,0,0]
		p1 = 0.039523999212758336
		p2 = 0.04080891538920051
		p3 = 0.14421632072267224
		p4 = 0.12908862310308516
		p5 = 0.14223306716801098

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

		# Make a guess based on states, if the image does not fall into a state category
		# then make a guess based on the confidence score
		# [0: "Anger", 1: "Disgust", 2: "Fear", 3: "Joy", 4: "Sadness", 5: "Surprise", 6: "Neutral"]
		emotion_guess = -1			  # Holds emotion guess
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
		else:						   # image does not fall into a state category
			emotion_guess = np.argsort(confidence)[-1]

		cv2.putText(frame, emotions[emotion_guess], (50,70), cv2.FONT_HERSHEY_SIMPLEX, 4, 255)

		cv2.imshow("Frame", frame)

		key = cv2.waitKey(1) & 0xFF

		if key == ord("q"):
			break

		elif key %256 == 32:   # SPACE pressed
			# get bounding box of face
			(x, y, w, h) = face_utils.rect_to_bb(faces)
			#cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			if w < h:
				w = h
			else:
				h = w
			im = gray[ y:y+h , x: x + w ]
			im = imutils.resize(im, width=48)

			#print(im.shape)

			d = emotion_guess
			uc = UnsupervisedClassification()
			u = uc.label_image([im])
			print("\nDeterministic:",emotions[d])
			print("Unsupervised:",emotions[u])
			asd = time.time()
			im = im.reshape((48, 48, 1))
			s = guess(im)
			print("Supervised:",emotions[s])
			# if s == d and d == u:
			# 	decision = s
			# elif s == u:
			# 	decision = s
			# elif s == d:
			# 	decision = s
			# elif d == u:
			# 	decision = d
			# else:
			# 	decision = s
			if d == u:
				decision = d
			else:
				decision = s
			print("Decision:",emotions[decision],"\n")
			# print("TIME",time.time() - asd)
			#img_name = "opencv_frame_{}.png".format(img_counter)
			#cv2.imwrite(img_name, frame)
			#print("{} written!".format(img_name))
			#img_counter += 1

def determinictic(frame):
	# Load facial predictor
	print("[INFO] loading facial landmark predictor...")
	shape_predictor_file = "shape_predictor_68_face_landmarks.dat"
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor_file)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	faces = detector(gray, 0)

	if len(faces) == 0:
		return -1
	elif len(faces) > 1:
		return -1
	else:
		faces = faces[0]

	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	landmarks = predictor(frame, faces)
	landmarks = shape_to_np(landmarks)

	#loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	for (x, y) in landmarks:
		cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

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

	# Calculate confidence scores of each emotion
	confidence = []

	# Certain characterictic distances seem to matter more,
	# therefore give them different weights if influence = True
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

	# Calculate confidence percentage based on how far each
	# characteristic distance is from the calculate average distances
	# for each emotion
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

	# Calculate distance from average neutral expressions
	L1 = D1 - AVG_D1_Neutral
	L2 = D2 - AVG_D2_Neutral
	L3 = D3 - AVG_D3_Neutral
	L4 = D4 - AVG_D4_Neutral
	L5 = D5 - AVG_D5_Neutral

	# Determine a "state" for L distances based on calculated
	# partions from -1 to 1
	state = [0,0,0,0,0]
	p1 = 0.039523999212758336
	p2 = 0.04080891538920051
	p3 = 0.14421632072267224
	p4 = 0.12908862310308516
	p5 = 0.14223306716801098

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

	# Make a guess based on states, if the image does not fall into a state category
	# then make a guess based on the confidence score
	# [0: "Anger", 1: "Disgust", 2: "Fear", 3: "Joy", 4: "Sadness", 5: "Surprise", 6: "Neutral"]
	emotion_guess = -1			  # Holds emotion guess
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
	else:						   # image does not fall into a state category
		emotion_guess = np.argsort(confidence)[-1]
	return emotion_guess
	

def evaluate():
	dataset_file = "./fer2013/fer2013.csv"
	emotions = ["Anger", "Disgust", "Fear",
				"Joy", "Sadness", "Surprise", "Neutral"]

	# open dataset
	csvr = csv.reader(open(dataset_file))
	# Get training and test set
	print("[INFO] loading dataset...")
	rows = [list(row) for row in csvr]
	data = pd.read_csv(dataset_file).values
	training_emotions = [data[i,0] for i in range(len(data)) if data[i,2] == 'Training']
	training_set = [np.fromstring(data[i,1], dtype=np.uint8, sep=' ') for i in range(len(data)) if data[i,2] == 'Training']
	test_emotions = [data[i,0] for i in range(len(data)) if data[i,2] == 'PublicTest' or data[i,2] == 'PrivateTest']
	test_set = [np.fromstring(data[i,1], dtype=np.uint8, sep=' ') for i in range(len(data)) if data[i,2] == 'PublicTest' or data[i,2] == 'PrivateTest']

	# reshape images into 48x48 image
	training_images = [i.reshape((48, 48)) for i in training_set]
	test_images = [i.reshape((48, 48)) for i in test_set]

	combination(training_images, training_emotions)
	
def combination(images, images_emotions):
	guesses = 0
	correct_guesses = 0
	for (i, image) in enumerate(images):
		# show loop progress
		print(i, "out of", len(images))

		uc = UnsupervisedClassification()
		u = uc.label_image([images])
		s = guess(images)

		if d == u:
			decision = d
		else:
			decision = s
		# Check if accurate guess
		if images_emotions[i] == decision:
			correct_guesses += 1
		guesses += 1

		# #show the output image with the face detections + facial landmarks
		# cv2.imshow("Output", image)

		# if cv2.waitKey(0) == ord('q'):
		#	 quit()
		# else:
		#	 continue

	# Check accuracy
	accuracy = 0
	accuracy = (correct_guesses / guesses) * 100
	print("Accuracy:", accuracy)



if __name__ == '__main__':
	#video_demo()
	evaluate()
	

