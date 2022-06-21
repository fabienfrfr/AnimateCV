#fabienfrfr - 20220621

import numpy as np

def pseudo_square(vector):
	size = len(vector)
	root = np.sqrt(size)
	# h,w
	h = int(np.rint(root))
	w = h
	while h*w > size :
		w -= 1
	# truncated vector
	trunc = vector[:h*w]
	return trunc.reshape((h,w))
