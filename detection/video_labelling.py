# fabienfrfr 20220708
import os, argparse, tqdm
import re, cv2

## Input
parser = argparse.ArgumentParser(description='Animate an image understanding by a neural network - App')
parser.add_argument('-i', '--input', type=str, required=False, help="Path of the image list to load.")

## Path
INPUT_DIR = 'output/'
OUTPUT_DIR = 'out_video/'

def description(video_name, frame):
	Text = video_name.split('.')[0].split('_')[1]
	# add description
	start_point, end_point = (0,0),(300,30)
	origin, fontScale, color, thickness = (20, 20), 0.5, (0, 0, 0, 255), 1
	text = re.sub('-',':',Text[0].upper() + Text[1:]) # sub('[a-z]')
	cv2.rectangle(frame, start_point, end_point, (255, 255, 255, 255), cv2.FILLED)
	cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA, False)

if __name__ == '__main__':
	args = parser.parse_args()
	if args.input == None :
		video_list = os.listdir(INPUT_DIR)
		video = []
		for i in range(len(video_list)) :
			if "video" in video_list[i][:5] : 
				video += [video_list[i]]
		video_list = video; del(video)
	print('[INFO] Starting System')
	## adding label rectangle for each video
	for v in video_list :
		video_capture = cv2.VideoCapture(INPUT_DIR + v)
		# get parameter
		size = (int(video_capture.get(3)), int(video_capture.get(4)))
		# opencv writer
		codec = cv2.VideoWriter_fourcc(*'mp4v')
		result = cv2.VideoWriter(OUTPUT_DIR + v ,codec, 25, size)
		while True:
			ret, frame = video_capture.read()
			if frame is None :
				break
			## adding description
			description(v, frame)
			## write frame
			result.write(frame)
			## show
			cv2.imshow('Video', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		# free intermediate video capture
		video_capture.release()
		result.release()
	print('[INFO] Stopping System')
	cv2.destroyAllWindows()