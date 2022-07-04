# fabienfrfr - 20220704
import numpy as np #, pylab as plt
import cv2

from utils_math import CenterCropping

## Color
colors = {'C0':'#1f77b4ff','C1':'#ff7f0eff','C2':'#2ca02cff','C3':'#d62728ff','C4':'#9467bdff'}

def hex_to_rvba(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], base=16) for i in range(0, lv, lv//3))

def stabilize_video(value):
	pass

def intermediate_layers_part(input_image, inter_layer, resizing):
	NB_LAYERS = len(inter_layer)
	merged = []
	for n in range(NB_LAYERS) :
		value = inter_layer[n][1]
		# normalize
		value = 255.*(value-value.min())/(value.max()-value.min())
		# resize layer t
		layer = cv2.resize(np.uint8(value), resizing, interpolation = cv2.INTER_AREA)
		# viridis
		layer = cv2.applyColorMap(layer, cv2.COLORMAP_VIRIDIS)
		# merging (gradient input to output)
		dst = cv2.addWeighted(input_image,1.0 - n/NB_LAYERS, layer,0.5 + (n/NB_LAYERS)/2.,0)
		merged += [dst[None][None]]
	return merged

def semantic_segmentation_part(input_image, head_norm, last_output, resizing, NB_OBJECT = 1):
	head = (head_norm - head_norm.min()) / (head_norm.max()- head_norm.min())
	out, norm = last_output + 1, 255*head
	# semantics & bg
	bg = out.copy()
	semantics, count = np.unique(out, return_counts=True)
	out[out >= semantics[count.argmax()]] -= 1
	bg[bg == semantics[count.argmax()]] = 0
	bg[norm < np.pi*norm.mean()] = 0
	bg[bg != 0] = 1
	# connected componnent
	_, labels_map = cv2.connectedComponents(np.uint8(bg))
	labels, count = np.unique(labels_map, return_counts=True)
	if NB_OBJECT > 0 :
		# nb object defined
		sorted_ = count.argsort()[::-1]
		for c in sorted_[NB_OBJECT+1:]:
			labels_map[labels_map == labels[c]] = 0
	else :
		# case nb_object undefined
		labels_noise = labels[count < np.percentile(count,75)]
		for noise in labels_noise :
			labels_map[labels_map == noise] = 0
	# find countour mask
	mask = labels_map
	mask[mask != 0] = 1
	contours, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# final image
	img = out.copy()
	img[mask == 0] = 0
	img = img.astype('object')
	idx = tuple(map(tuple,np.where(img == 0)))
	img[idx] = [(0,0,0)]
	for k,v in colors.items() :
		idx = tuple(map(tuple,np.where(img == int(k[1])+1)))
		img[idx] = [hex_to_rvba(v)[:3]]
	frame = np.uint8(np.array(img.tolist()))
	frame = cv2.resize(frame, resizing, interpolation = cv2.INTER_AREA)
	#frame = cv2.GaussianBlur(frame,(15,15),0)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	#cv2.drawContours(frame, contours, -1, (0,255,0), 3)
	frame[frame == 0] = input_image[frame == 0]
	
	return frame[None][None]