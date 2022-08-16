# fabienfrfr - 20220628
## Import package
import sys
v,vv,_,_,_ = sys.version_info
print('[INFO] Python version : ' + str(v) + '.' + str(vv))

import numpy as np, pylab as plt
import os, argparse, logging, copy
import cv2, torch

import torch.nn as nn

import utils
import vision_transformer as vits

from utils_math import CenterCropping

import hub # old pytorch

## Input
parser = argparse.ArgumentParser(description='Animate an image understanding by a neural network - App')
parser.add_argument('-i', '--input', type=str, required=False, help="Path of the image to load.")
parser.add_argument('-imz', '--image_size', default=(720, 720), type=int, nargs="+", help="Resize image.")

## Path
INPUT_DIR = 'input/'
OUTPUT_DIR = 'output/'

def ViT_dino_model(info=True):
	if info : logging.root.setLevel(logging.INFO)
	else : logging.root.setLevel(logging.WARNING)
	## Variable affectation
	PATCH_SIZE = 8
	MODEL_NAME = 'vit_small'
	## Neural network Initialization
	logging.info("Starting System...")
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	logging.info("Calculation type : " + device.type)
	logging.info('Importing build torch model (k=8x8)..')
	model = vits.__dict__[MODEL_NAME](patch_size=PATCH_SIZE, num_classes=0)
	logging.info('Switch models to inference mode..')
	for p in model.parameters(): 
		p.requires_grad = False
	model.eval(); model.to(device)
	logging.info('Importing pretrained dino model..')
	url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth" # model used for visualizations in dino paper
	state_dict = hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
	model.load_state_dict(state_dict, strict=True)
	logging.info('Torch model ready to use !')
	return model, device

# self supervised detection
def process_ssd(image, model, device, img_resize, info=True):
	if info : logging.root.setLevel(logging.INFO)
	else : logging.root.setLevel(logging.WARNING)
	logging.info('Define checkpoint of intermediate result layer..')
	fhooks, vit, name = [], [], []
	for i,l in enumerate(list(model._modules.keys())):
		if l == 'blocks' :
			logging.info('Extract attention of each blocks output:(x,attn)..')
			for j, k in enumerate(list(model.blocks._modules.keys())) :
				name += [copy.deepcopy(l) + '-' + copy.deepcopy(k)]
				model.blocks[j]._modules['attn'].register_forward_hook(lambda m, input, output: vit.append((output[1])))
		## afterwards, to avoid calculating the last block twice
		name += [copy.deepcopy(l)]
		fhooks.append(getattr(model,l).register_forward_hook(lambda m, input, output: vit.append((output))))
	logging.info('Adapt image for dnn model..')
	blob = cv2.dnn.blobFromImage(image, 1/255, img_resize, (0,0,0), swapRB=True, crop=False)
	img = torch.tensor(blob)
	w_featmap = img.shape[-2] // model.PATCH_SIZE
	h_featmap = img.shape[-1] // model.PATCH_SIZE
	logging.info('Processing neural network last attentions..')
	attentions = model.get_last_selfattention(img.to(device))
	logging.info('Attention size is : ' + str(attentions.shape))
	nh = attentions.shape[1] # number of head
	logging.info('Keep only the output patch attention..')
	attention = attentions[0, :, 0, 1:].reshape(nh, -1) # dim:2 == 0, is output patch attention
	logging.info('Reshape..')
	attention = attention.reshape(nh, w_featmap, h_featmap)
	#attention = nn.functional.interpolate(attention.unsqueeze(0), scale_factor=model.PATCH_SIZE, mode="nearest")[0]
	attention = attention.cpu().numpy()
	logging.info('Saving checkpoint..')
	vit = [(n,out) for (n,out) in zip(name,vit)]
	logging.info('Processing done !')
	# free memory 
	del fhooks, model
	return attention, vit

def extract_vit_feature(vit, last_attention, img_size, patch_size, head=False, max=False, info=True):
	if info : logging.root.setLevel(logging.INFO)
	else : logging.root.setLevel(logging.WARNING)
	layers_out = []
	# mean pooling for each layers
	for v in vit :
		logging.info('Extract feature of ' + v[0])
		## before block
		if v[1].dim() == 3 :
			if v[1].shape[1] % 2 == 0 :
				out = v[1].squeeze().mean(axis=1)
			# dropout
			else :
				out = v[1].squeeze().mean(axis=1)[1:]
		## attention block (after dropout)
		elif v[1].dim() == 4 :
			# max pooling for head (or sum)
			if max : out,_ = torch.max(v[1].squeeze(), dim=0)
			else : out = torch.sum(v[1].squeeze(), dim=0)
			out = out[0][1:]
		## reshaping and scaling
		out = out.reshape(img_size[0] // patch_size, img_size[1] // patch_size)
		## save
		layers_out += [[v[0], out.cpu().numpy()]]
	# all head
	if head : layers_out += [["last_output-head"+str(i), last_attention[i]] for i in range(last_attention.shape[0])]
	# argmax last attention head indice (self-semantique segmentation)
	layers_out += [["last_output", last_attention.argmax(axis=0)]]
	# free memory 
	del vit, last_attention
	return layers_out

if __name__ == '__main__':
	## import model
	model, device = ViT_dino_model()
	args = parser.parse_args()
	## exemple
	if args.input == None :
		img_list = os.listdir(INPUT_DIR)
		img_name = np.random.choice(img_list)
	else :
		img_name = args.input
	image_path = INPUT_DIR + img_name
	print('[INFO] Importing image ' + img_name + '..')
	image = cv2.imread(image_path)
	image = CenterCropping(image) #not necessary
	# apply
	att, vit = process_ssd(image, model, device, args.image_size)
	print('[INFO] Show result..')
	layers_out = extract_vit_feature(vit, att)
	for l in layers_out :
		print('[INFO] ' + l[0])
		plt.imshow(l[1]); plt.show()
		plt.imsave(fname= OUTPUT_DIR+l[0], arr=l[1], format='png')
	print('[INFO] Stopping System')


"""
import skvideo.io as skio
import numpy as np, cv2

video_attention = 255*(all_last_attention/all_last_attention.max())
skio.vwrite('test_att.mp4', video_attention)
"""
