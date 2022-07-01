# fabienfrfr - 20220628
import os, argparse, tqdm
import numpy as np, pylab as plt
import cv2, torch
import skvideo.io as skio

from models_import import ViT_dino_model, process_ssd

## Input
parser = argparse.ArgumentParser(description='Animate an image understanding by a neural network - App')
parser.add_argument('-i', '--input', type=str, required=False, help="Path of the image list to load.")
parser.add_argument('-imz', '--image_size', default=(240, 240), type=int, nargs="+", help="Resize image.")

## Path
INPUT_DIR = 'video/'
OUTPUT_DIR = 'output/'

@torch.no_grad()
def extract_ssd_on_davis():
	pass

if __name__ == '__main__':
	## import model
	model, device = ViT_dino_model()
	args = parser.parse_args()
	## exemple
	if args.input == None :
		img_list = sorted(os.listdir(INPUT_DIR), key = lambda im: im.split('.')[0])
	print('[INFO] Prepare images loop analysis..')
	video = []
	for t in tqdm.tqdm(range(len(img_list))) :
		image_path = INPUT_DIR + img_list[t]
		# print('[INFO] Importing image..')
		image = cv2.imread(image_path)
		# apply
		attentions, attention, vit = process_ssd(image, model, device, args.image_size, info=False)
		video += [np.sum(attention,axis=0)[None]]
	print('[INFO] Convert img list to video..')
	video_ = 255*(np.concatenate(video)-np.min(video))/(np.max(video)-np.min(video))
	video_[video_ < 2*video_.mean()] = 0
	print('[INFO] Writting video..')
	writer = skio.FFmpegWriter(OUTPUT_DIR + "outputvideo.mp4")
	for v in video_:
		#frame = cv2.cvtColor(v.astype(np.uint8), cv2.COLOR_GRAY2BGR)
		frame = cv2.applyColorMap(v.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
		#frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
		writer.writeFrame(frame)
	writer.close()
	# 
	cv2.imshow('viridis_frame', frame)
	cv2.waitKey(0) & 0xFF == ord('q')
	cv2.destroyAllWindows()
	print('[INFO] Stopping System')

"""
import skvideo.io as skio
import numpy as np, cv2

vid = skio.vread("outputvideo_480.mp4")
median = np.median(vid)
vid[vid < 3*vid.mean() + 3*median] = 0
skio.vwrite("outputvideo.mp4", vid)
"""