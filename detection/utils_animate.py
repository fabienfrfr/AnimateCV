# fabienfrfr - 20220704
import numpy as np #, pylab as plt
import cv2

from utils_math import CenterCropping

## Color (nb of head)
colors = {'C0':'#1f77b4ff','C1':'#ff7f0eff','C2':'#2ca02cff','C3':'#d62728ff','C4':'#9467bdff', 'C5': '#8c564bff'}

def hex_to_rvba(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv//3], base=16) for i in range(0, lv, lv//3))

def rgba_blending(img_seq, color = '#ffffffff', thresh = 0.5):
	c = hex_to_rvba(color)
	f = np.vectorize(lambda x : 1/(1+np.exp(-10*(x-thresh))))
	s = f(img_seq[:,:,:,3]/255.)
	rgb_blend = []
	for i in range(3) :
		rgb_blend += [(((1 - s) * c[i]) + (s * img_seq[:,:,:,i]))[:,:,:,None]]
	rgb_blend = np.concatenate(rgb_blend, axis=3)
	return np.uint8(rgb_blend)

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
	# binarization
	alpha = cv2.normalize(head_norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

	bin_img = cv2.GaussianBlur(alpha,(5,5),0)
	bin_img = cv2.resize(bin_img, resizing, interpolation = cv2.INTER_AREA)
	bin_img = cv2.GaussianBlur(bin_img,(15,15),0)

	_, thresh = cv2.threshold(bin_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	# connected component
	_, mask = cv2.connectedComponents(thresh)
	if NB_OBJECT == 1 :
		labels, count = np.unique(mask[mask != 0], return_counts=True)
		mask[mask != labels[count.argmax()]] = 0
	contours, hierarchy = cv2.findContours(np.uint8(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# Extract semantics
	semantic_img = np.uint8(last_output + 1)
	semantic_img = cv2.resize(semantic_img, resizing, interpolation = cv2.INTER_AREA)
	semantic_img[mask == 0] = 0
	# apply semantic
	image = semantic_img.astype('object')
	for i in np.unique(image):
		idx = tuple(map(tuple,np.where(image == i)))
		if i == 0 :
			image[idx] = [(255,255,255)]
		else :
			c = list(colors.values())[i-1]
			image[idx] = [hex_to_rvba(c)[:3]]
	image = np.array(image.tolist())
	image = cv2.GaussianBlur(np.uint8(image), (15,15), 0)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# final image
	image[image == 255] = input_image[image == 255]
	image = cv2.addWeighted(input_image,1.0, image,0.8,0)
	#cv2.drawContours(image, contours, -1, (0,255,0), 3)
	## simple alpha image
	simple = cv2.normalize(last_output, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	simple = cv2.resize(simple, resizing, interpolation = cv2.INTER_AREA)
	simple = cv2.applyColorMap(simple, cv2.COLORMAP_VIRIDIS)
	simple = cv2.cvtColor(simple, cv2.COLOR_RGB2BGR) # for matplotlib
	alpha = cv2.resize(alpha, resizing, interpolation = cv2.INTER_AREA)
	alpha[mask == 0] = alpha[mask == 0] /10.
	simple = np.concatenate((simple, np.uint8(alpha[:,:,None])), axis=2)
	return image[None][None], simple[None]

