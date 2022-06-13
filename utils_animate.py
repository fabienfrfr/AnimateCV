# fabienfrfr - 20220613

import cv2

def plot(inter_out):
	#image = cv2.imread(image_path)
	#image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	# Window name in which image is displayed
	window_name = 'Image'
	  
	# text
	text = 'GeeksforGeeks'
	  
	# font
	font = cv2.FONT_HERSHEY_SIMPLEX
	  
	# org
	org = (00, 185)
	  
	# fontScale
	fontScale = 1
	   
	# Red color in BGR
	color = (0, 0, 255)
	  
	# Line thickness of 2 px
	thickness = 2

	for i,array in inter_out :
		imax = 0
		image = array[0][imax]
		image = cv2.putText(image, text, org, font, fontScale, color, thickness, cv2.LINE_AA, False)
		cv2.imshow(window_name, image)
'''
import itertools

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, cm as cm
import skimage
import skimage.transform

def visualize_att(image_path, seq, alphas, idx2word, endseq='<end>', smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param idx2word: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """

    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    # words = [idx2word[ind] for ind in seq]
    words = list(itertools.takewhile(lambda word: word != endseq, map(lambda idx: idx2word[idx], iter(seq))))

    for t in range(len(words)):
        if t > 50:
            break

        index = int(np.ceil(len(words) / 5.))
        plt.subplot(index, 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    plt.show()
'''