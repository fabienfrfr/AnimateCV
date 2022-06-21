# fabienfrfr - 20220618
import numpy as np, pylab as plt
import cv2, dlib
import os, argparse

from utils_math import pseudo_square

## Input
parser = argparse.ArgumentParser(description='Animate an image understanding by a neural network - App')
parser.add_argument('-i', '--input', type=str, required=False, help='directory of input known faces')

## Path
MODEL_DIR = 'saved_models/'
INPUT_DIR = 'input/'

## global param
DEFAULT_CONFIANCE = 0.15
THRESHOLD = 0.1

def detection_model(MODEL_DIR):
	print("[INFO] Importing pretrained YOLO model with COCO dataset (Tiny Darknet)..")
	bbox_detection = cv2.dnn.readNet(MODEL_DIR+'yolo/yolov3-tiny_thresh0.cfg', MODEL_DIR+'yolo/yolov3-tiny.weights') # Bounding box
	print("[INFO] Importing frontal face detector model (HOG+SVM)..")
	face_detector = dlib.get_frontal_face_detector()
	print("[INFO] Importing landmarks detector model (dataset ?)..")
	pose_predictor_68_point = dlib.shape_predictor(MODEL_DIR+"landmarks/shape_predictor_68_face_landmarks.dat")
	print("[INFO] Importing face recognition model (dataset ?)..")
	face_encoder = dlib.face_recognition_model_v1(MODEL_DIR+"face_recognition/dlib_face_recognition_resnet_model_v1.dat")
	# in logic order of execution
	return bbox_detection, face_detector, pose_predictor_68_point, face_encoder

def bounding_box_yolo(image,model,param):
	image_bb = image.copy()
	# parameter
	ln, labels = param
	height, width, _ = image.shape
	# apply (adapt before image for net & we have 2 detection layers for this networks)
	blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
	model.setInput(blob)
	layerOutputs = model.forward(ln)
	
	# initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
	grids, boxes, confidences, classIDs = [], [], [], []

	# loop over each of the layer outputs
	for output in layerOutputs:
		layer = output[:,5:]
		#layer[layer < 0.25] = 0.
		lbl_idx = np.argmax(layer,axis=1)
		grid = pseudo_square(lbl_idx).astype('uint8').T
		grids += [cv2.resize(grid, (height, width), interpolation = cv2.INTER_AREA).T]
		# loop over each of the detections
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# select & scale
			if confidence > DEFAULT_CONFIANCE:
				box = detection[0:4] * np.array([width, height, width, height])
				(centerX, centerY, W, H) = box.astype("int")
				x = int(centerX - (W / 2))
				y = int(centerY - (H / 2))
				# update
				boxes.append([x, y, int(W), int(H)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping bounding box
	indexes = cv2.dnn.NMSBoxes(boxes, confidences, DEFAULT_CONFIANCE, THRESHOLD)
	# initialize a list of colors to represent each possible class label
	COLORS = 255*plt.cm.get_cmap('viridis',len(labels)).colors[:,:3]

	# ensure at least one detection exists
	if len(indexes) > 0:
		# loop over the indexes we are keeping
		for i in indexes.flatten():
			# extract the bounding box coordinates
			(x, y, w, h) = boxes[i]
			# draw a bounding box rectangle and label on the frame
			color = COLORS[classIDs[i]]
			text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
			cv2.rectangle(image_bb, (x, y), (x + w, y + h), color, 2)
			cv2.putText(image_bb, text, (x, y + 20 ), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
	
	return image_bb, grids

def landmark_detection(frame, face_detector, pose_predictor_68_point):
	rgb_small_frame = frame[:, :, ::-1]
	face_locations = face_detector(rgb_small_frame, 1)
	shapes, landmarks_list = [],[]
	# face detection loop with landmark
	for d in face_locations :
		cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), 255, 1)
		shape = pose_predictor_68_point(frame, d)
		shapes += [shape]
		shape = [(i.x, i.y) for i in shape.parts()]
		landmarks_list += [shape]
	# add landmarks
	for shape in landmarks_list:
			for (x, y) in shape:
				cv2.circle(frame, (x, y), 1, (255, 0, 255), -1)
	return landmarks_list, shapes

if __name__ == '__main__':
	print("[INFO] Starting System...")
	## import model
	bbox_detection, face_detector, pose_predictor_68_point, face_encoder = detection_model(MODEL_DIR)
	args = parser.parse_args()

	print('[INFO] Prepare model..')
	layer_name_output = [bbox_detection.getLayerNames()[i - 1] for i in bbox_detection.getUnconnectedOutLayers()]
	print('[INFO] Importing coco class labels..')
	with open(MODEL_DIR + 'yolo/coco.names', 'r') as f:
		LABELS = f.read().splitlines()
	print('[INFO] Importing face model..')
	if args.input == None :
		known_face_files = os.listdir(INPUT_DIR)
	else :
		known_face_files = os.listdir(args.input)

	print('[INFO] Starting Webcam...')
	video_capture = cv2.VideoCapture(0)
	print('[INFO] Webcam well started')
	print('[INFO] Detecting...')
	while True:
		ret, frame = video_capture.read()
		#cv2.imshow('Input camera', frame)
		## YOLO
		img_bb, grids = bounding_box_yolo(frame,bbox_detection,(layer_name_output, LABELS))
		"""
		dst = cv2.addWeighted(frame,1.0,img_rgb,0.8,0)
		"""
		cv2.imshow('Object detection i', img_bb) # dst) #obj_img)
		## FACE
		landmarks_list, shapes = landmark_detection(frame, face_detector, pose_predictor_68_point)
		cv2.imshow('Landmarks detection with frontal face detection (i,j)', frame)
		face_encodings_list = []
		for s in shapes :
			face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(frame, s, num_jitters=1)))
		for face_encoding in face_encodings_list:
			#vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
			pass
		"""
		cv2.imshow('Face recognition (i,k)', frame)
		cv2.imshow('Composite', frame)
		"""
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	print('[INFO] Stopping System')
	video_capture.release()
	cv2.destroyAllWindows()