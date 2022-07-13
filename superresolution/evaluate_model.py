# fabienfrfr - 20220709
## Import package
import numpy as np, pylab as plt
import torch, cv2
import argparse, os

## Input
parser = argparse.ArgumentParser(description='Animate an image understanding by a neural network - App')
parser.add_argument('-i', '--input', type=str, required=False, help='directory of input image')

## Path
MODEL_DIR = 'saved_model/'
MODEL_NAME = "checkpoint_srgan.pth.tar"
INPUT_DIR = 'input/'
OUTPUT_DIR = 'output/'

## function
def srgan_generator_model(MODEL_DIR,MODEL_NAME):
	## Variable affectation

	## Neural network Initialization
	print("[INFO] Starting System...")
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print("[INFO] Calculation type : " + device.type)
	print('[INFO] Importing pretrained model..')
	checkpoint = torch.load(MODEL_DIR+MODEL_NAME, map_location=device)['generator']
	print('[INFO] Switch models to inference mode..')
	checkpoint.eval()
	model = checkpoint
	print('[INFO] Torch model ready to use !')
	return model

## run
if __name__ == '__main__':
	## import model
	model = srgan_generator_model(MODEL_DIR,MODEL_NAME)
	args = parser.parse_args()
	## exemple
	print('[INFO] Testing model')
	if args.input == None :
		img_list = os.listdir(INPUT_DIR)
		img_name = np.random.choice(img_list)
	else :
		img_name = args.input
	print('[INFO] Image choised :' + img_name)
	image_path = INPUT_DIR + img_name
	print('[INFO] Computation without gradient (memory)')
	# Prohibit gradient computation explicitly because I had some problems with memory
	with torch.no_grad():
		print('[INFO] Importing image')
		image = cv2.imread(image_path)
		plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)); plt.show()
		print('[INFO] Converting image')
		blob = cv2.dnn.blobFromImage(image, 1/255, image.shape[:2], (0,0,0), swapRB=True, crop=False)
		lr_imgs = torch.tensor(blob)
		print('[INFO] Apply image in model')
		out = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]
		img = np.moveaxis(out[0].cpu().numpy(), 0,2)
		img = cv2.normalize(img, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		# show result
		plt.imshow(img); plt.show()
		# save
		img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		cv2.imwrite(OUTPUT_DIR + img_name.split('_')[0] + '_hr.jpg', img_bgr) 