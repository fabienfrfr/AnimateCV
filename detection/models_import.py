# fabienfrfr - 20220618
import numpy as np, pylab as plt
import cv2, dlib
import os, argparse

## Input
parser = argparse.ArgumentParser(description='Animate an image understanding by a neural network - App')
parser.add_argument('-i', '--input', type=str, required=False, help='directory of input known faces')

## Path
MODEL_DIR = 'saved_models/'
INPUT_DIR = 'input/'

def detection_model(MODEL_DIR):
	print("[INFO] Importing pretrained YOLO model with COCO dataset and five scales architechtre network (Darknet)..")
	bbox_detection = cv2.dnn.readNet(MODEL_DIR+'yolo/yolov4.cfg', MODEL_DIR+'yolo/yolov4.weights') # Bounding box
	print("[INFO] Importing frontal face detector model (HOG+SVM)..")
	face_detector = dlib.get_frontal_face_detector()
	print("[INFO] Importing landmarks detector model (dataset ?)..")
	pose_predictor_68_point = dlib.shape_predictor(MODEL_DIR+"landmarks/shape_predictor_68_face_landmarks.dat")
	print("[INFO] Importing face recognition model (dataset ?)..")
	face_encoder = dlib.face_recognition_model_v1(MODEL_DIR+"face_recognition/dlib_face_recognition_resnet_model_v1.dat")
	# in logic order of execution
	return bbox_detection, face_detector, pose_predictor_68_point, face_encoder

def bounding_box_yolo(image,model,param):
	# parameter
	ln, labels = param
	# apply (adapt before image for net)
	blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
	model.setInput(blob)
	layerOutputs = model.forward(ln)
	# object detection (we have 3 detection layers for this networks)


if __name__ == '__main__':
	print("[INFO] Starting System...")
	## import model
	bbox_detection, face_detector, pose_predictor_68_point, face_encoder = detection_model(MODEL_DIR)
	args = parser.parse_args()

	print('[INFO] Prepare model..')
	layer_name_output = [bbox_detection.getLayerNames()[i - 1] for i in bbox_detection.getUnconnectedOutLayers()]
	output_before_relu = [bbox_detection.getLayerNames()[i - 3] for i in bbox_detection.getUnconnectedOutLayers()]
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
		cv2.imshow('Input camera', frame)
		obj_img = frame
		blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
		bbox_detection.setInput(blob)
		layerOutputs = bbox_detection.forward(layer_name_output)
		Outputs_before = bbox_detection.forward(output_before_relu)
		conv_max = np.argmax(Outputs_before[1][0],axis=0).astype('uint8')
		img = cv2.resize(conv_max, frame.shape[:2], interpolation = cv2.INTER_AREA).T
		img_norm_8U = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		img_rgb = cv2.cvtColor(img_norm_8U,cv2.COLOR_GRAY2RGB)
		dst = cv2.addWeighted(frame,1.0,img_rgb,0.8,0)
		cv2.imshow('Object detection i', dst) #obj_img)
		"""
		cv2.imshow('Frontal face detection (i,j)', frame)
		cv2.imshow('Landmarks detection (i,j)', frame)
		cv2.imshow('Face recognition (i,k)', frame)
		cv2.imshow('Composite', frame)
		"""
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	print('[INFO] Stopping System')
	video_capture.release()
	cv2.destroyAllWindows()