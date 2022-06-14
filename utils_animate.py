# fabienfrfr 20220614
import os, cv2
import skimage as skim
import numpy as np, pylab as plt
import matplotlib.animation as animation

from utils_math import GRID_GEN
"""
Image imported by opencv, the format is RGB, but the model trained with BGR format 
and matplotlib show BGRA type of image encoding.
"""

def show_all_channel(inter_out, DIM=(720,720)):
	# Loop for each layers (width, height)
	for name,array in inter_out :
		shape = array.numpy()[0].shape
		grid = GRID_GEN(shape[0])[1]
		print('[INFO] Layer : '+str(name)+'; Shape : '+str((shape,grid)))
		img_complete = array.numpy().reshape((shape[1]*grid[0], shape[2]*grid[1]))
		img_complete = cv2.resize(img_complete, DIM, interpolation = cv2.INTER_AREA)
		cv2.imshow("All channel of Layers : "+str(name), img_complete)
		cv2.waitKey(0) == ord('q')
		# free memory
		cv2.destroyAllWindows()

def image_construct(img, input_, name, Text, smooth=False, alpha=True, DIM=(720,720)):
	# MinMax Normalization 1
	img = (img-img.min())/(img.max()-img.min())
	if smooth :
		img = skim.transform.pyramid_expand(img, upscale=6, sigma=2)
	# Resizing with slight blur
	img = cv2.resize(img, DIM, interpolation = cv2.INTER_AREA)
	img = cv2.GaussianBlur(img,(5,5),0)
	# Float to 8bit (0-255)
	img = (img-img.min())/(img.max()-img.min())
	img_norm_8U = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	# Alpha or Superpose : destination image
	if alpha :
		input_bgra = cv2.cvtColor(input_,cv2.COLOR_RGB2RGBA)
		dst = input_bgra.copy()
		dst[:,:,3] = img_norm_8U
	else :
		img_rgb = cv2.cvtColor(img_norm_8U,cv2.COLOR_GRAY2RGB)
		dst = cv2.addWeighted(input_,1.0,img_rgb,0.8,0)
	# add description
	start_point, end_point = (0,0),(300,30)
	origin, fontScale, color, thickness = (20, 20), 0.5, (0, 0, 0, 255), 1
	text = Text[0] + str(name) + Text[1]
	cv2.rectangle(dst, start_point, end_point, (255, 255, 255, 255), cv2.FILLED)
	cv2.putText(dst, text, origin, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, cv2.LINE_AA, False)
	return dst

def construct_arrayAnimation(image_input, inter_out, traitment_type, custom_txt, fpl = 6, DIM=(720,720),smooth=False, alpha=True):
	### Text parameter
	Text = custom_txt
	print('[INFO] Transform input image..')
	input_resize = cv2.resize(image_input, DIM, interpolation = cv2.INTER_AREA)
	if traitment_type == 0 :
		# Loop for each layers (frame per layers)
		input_bgra = cv2.cvtColor(input_resize,cv2.COLOR_RGB2RGBA)
		arrayList = [input_bgra.copy()]
		for name,array in inter_out :
			# Define high level of output response
			idxmax = np.argsort(array[0].sum(axis=(1,2)).numpy())[::-1]
			print('[INFO] Extract best feature of layer : '+str(name))
			for idx in idxmax[:fpl]:
				img = array[0].numpy()[idx]
				arrayList += [image_construct(img,input_resize, name, Text).copy()]
	else :
		print('[INFO] Extract Attention : ')
		arrayList = []
		for name,array in inter_out :
			img = array[0].numpy()
			arrayList += fpl*[image_construct(img, input_resize, name, Text, smooth, alpha).copy()]
	print('[INFO] Feature extracted, ready for animation !')
	return arrayList

class animate_2dArray():
	def __init__(self, Array_list, Figsize=(10,10), DPI=60):
		self.imArray = Array_list
		# init axes
		self.fig = plt.figure(figsize=Figsize, dpi=DPI) 
		self.ax = self.fig.add_subplot(111)
		self.im = plt.imshow(self.imArray[0])
		# edit
		self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
		plt.xticks([]), plt.yticks([])
		# Delete border
		self.ax.spines['bottom'].set_color('None'); self.ax.spines['top'].set_color('None') 
		self.ax.spines['right'].set_color('None'); self.ax.spines['left'].set_color('None')

	def animation_step(self, i):
		self.im.set_array(self.imArray[i])
		return self.im,

	def animate(self, name = "output"):
		self.anim = animation.FuncAnimation(self.fig, self.animation_step, frames=len(self.imArray), blit=False, interval=1)
		self.anim.save(filename= 'output' + os.path.sep + name +'.mp4', writer='ffmpeg', fps=6) # png for alpha