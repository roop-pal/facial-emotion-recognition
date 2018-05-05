# import the necessary packages
from imutils import face_utils
import csv
import numpy as np
#import argparse
from matplotlib import pyplot as plt
import imutils
import dlib
import cv2

def shape_to_np(shape, dtype="int"):
	# convert dlib shape to np array 
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# convert the 68 facial landmarks to a 2-tuple 
	# of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords


def main():
	# initialize constants
	shape_predictor_file = "shape_predictor_68_face_landmarks.dat"
	dataset_file = "../fer2013/fer2013.csv"

	# open dataset
	csvr = csv.reader(open(dataset_file))

	header = next(csvr)
	
	rows = [list(row) for row in csvr]

	# get training images
	training_set = [row[1:-1] for row in rows if row[-1] == 'Training']

	# transform training_set into 48 x 48 images
	training_images = []

	for i in training_set:
		# print (emotions[int(i[0])])
		tmp_image = np.fromstring(i[0], dtype=np.uint8, sep=" ").reshape((48, 48))

		training_images.append(tmp_image)

	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	# see https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(shape_predictor_file)

	for image in training_images:
		# detect faces in the grayscale image
		image = imutils.resize(image, width=500)
		faces = detector(image, 1)

		if len(faces) == 0:
			print("No faces found")
			continue
		elif len(faces) > 1:
			print("Multiple faces found")
			continue
		else:
			faces = faces[0]

		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(image, faces)
		shape = shape_to_np(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		#(x, y, w, h) = face_utils.rect_to_bb(face)
		#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

		# show the output image with the face detections + facial landmarks
		
		cv2.imshow("Output", image)
		cv2.waitKey(0)

if __name__ == "__main__":
	main()