# fabienfrfr - 20220628
import os, argparse, tqdm, tracemalloc
import numpy as np, pylab as plt
import cv2, torch
import skvideo.io as skio
import skimage as skim

from models_import import ViT_dino_model, process_ssd, extract_vit_feature
from utils_math import CenterCropping
from utils_animate import intermediate_layers_part, semantic_segmentation_part, rgba_blending

## Input
parser = argparse.ArgumentParser(description='Animate an image understanding by a neural network - App')
parser.add_argument('-i', '--input', type=str, required=False, help="Path of the image list to load.")
parser.add_argument('-imz', '--image_size', default=(720, 720), type=int, nargs="+", help="Resize image.")

## Path
INPUT_DIR = 'video/'
OUTPUT_DIR = 'output/'

@torch.no_grad() # memory problems... decompose
def extract_dino(image_path, args):
	## import model (find solution to delete hook memory leak) with no grad ?
	model, device = ViT_dino_model(info=False)
	# print('[INFO] Importing image..')
	image = cv2.imread(image_path)
	image = CenterCropping(image)
	# apply
	attention, vit = process_ssd(image, model, device, args.image_size,  info=False)
	# extract feature
	layers = extract_vit_feature(vit, attention, args.image_size, model.PATCH_SIZE, info=False)
	# free memory
	del model, device, image, attention, vit
	# save feature
	"""pickle"""
	return layers

def frame_construction(layers, image_path, resizing=(720,720)):
	## Image input
	image = cv2.imread(image_path)
	image = CenterCropping(image)
	input_image = cv2.resize(np.uint8(image), resizing, interpolation = cv2.INTER_AREA)
	name = ['input-image'] + [l[0].replace('_','-') for l in layers] + ['semantic-out']
	## Intermediate layers part
	inter_video = intermediate_layers_part(input_image, layers, resizing)
	## Output part..
	out_video, simple_out = semantic_segmentation_part(input_image, layers[-2][1], layers[-1][1], resizing)
	## merging
	frame = np.concatenate([input_image[None][None]]+inter_video+[out_video], axis=0)
	del layers, input_image, inter_video, out_video
	return frame, name, simple_out

if __name__ == '__main__':
	tracemalloc.start()
	args = parser.parse_args()
	## exemple
	if args.input == None :
		img_list = sorted(os.listdir(INPUT_DIR), key = lambda im: im.split('.')[0])
	#img_list = img_list[:2]
	print('[INFO] Stabilize video before crop (not yet necessary)..')
	# https://github.com/krutikabapat/Video-Stabilization-using-OpenCV
	print('[INFO] Images loop analysis..')
	video, simple_out = [], []
	for t in tqdm.tqdm(range(len(img_list))) :
		image_path = INPUT_DIR + img_list[t]
		# dino layer
		dino = extract_dino(image_path, args)
		# to video
		frame, name, out = frame_construction(dino, image_path)
		video += [frame]
		simple_out += [out]
		# free memory
		del dino
	video, simple_out = np.concatenate(video, axis=1), np.concatenate(simple_out, axis=0)
	print('[INFO] Writting video..')
	simple_blending = rgba_blending(simple_out, "#00000000", 0.33) # "#f2f2f2ff" or "#e6e6e6ff"
	skio.vwrite(OUTPUT_DIR + "video_self-semantic.mp4", simple_blending)
	#np.save('out_video/simple_out.npy', simple_out) # simple = np.load('out_video/simple_out.npy')
	#np.save('out_video/all.npy', video) # video = np.load('out_video/all.npy')
	"""
	for i in range(simple_out.shape[0]) :
		skim.io.imsave('out_video/simple_out_'+str(i)+'.png',simple_out[i])
	"""
	for i in range(video.shape[0]) :
		writer = skio.FFmpegWriter(OUTPUT_DIR + "video_"+name[i]+".mp4")
		for v in video[i] :
			frame_ = cv2.cvtColor(v, cv2.COLOR_RGB2BGR)
			writer.writeFrame(frame_)
		writer.close()
	print('[INFO] Stopping System')
	"""
	snapshot = tracemalloc.take_snapshot()
	top_stats = snapshot.statistics('traceback')

	# pick the biggest memory block
	stats = top_stats[:3]
	for stat in stats :
		print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
		for line in stat.traceback.format():
			print(line)
	# free memory
	del video, frame, writer
	"""