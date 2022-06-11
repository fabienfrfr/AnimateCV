# fabienfrfr - 20220610
## Import package
import numpy as np, pylab as plt
import os, pickle, itertools, argparse
import cv2, torch

from models.torch.resnet101_attention import Captioner

## Input
parser = argparse.ArgumentParser(description='Animate an image understanding by a neural network - App')
parser.add_argument('-i', '--input', type=str, required=False, help='directory of input known faces')

## Path
MODEL_DIR = 'saved_models/resnet101_attention-embed_lstm/'
MODEL_NAME = 'resnet101_attention-embed_lstm_best-train.pt'
INPUT_DIR = 'input/'

## Variable affectation
EMBEDDING_DIM, ATTENTION_DIM = 300, 256
DECODER_SIZE, BATCH_SIZE = 256, 16

## Neural network Initialization
print("[INFO] Starting System...")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("[INFO] Calculation type : " + device.type)
print('[INFO] Importing vocabulary..')
vocab_set = pickle.load(open(MODEL_DIR+'vocab_set.pkl', 'rb')) if os.path.exists(MODEL_DIR+'vocab_set.pkl') else None
vocab, word2idx, idx2word, max_len = vocab_set
vocab_size = len(vocab)
print('[INFO] Importing pretrained model (Datasets : ImageNet for encoder, Flicker8k for decoder)..')
checkpoint = torch.load(os.path.join(MODEL_DIR, MODEL_NAME), map_location=device)
print('[INFO] Importing pretrained success')
print('[INFO] Importing torch model..')
model = Captioner(encoded_image_size=14, encoder_dim=2048, attention_dim=ATTENTION_DIM, embed_dim=EMBEDDING_DIM, decoder_dim=DECODER_SIZE, vocab_size=vocab_size,).to(device)
if False : print(model.eval()) # show model details
print('[INFO] Affect state in torch model..')
model.load_state_dict(checkpoint['state_dict'])
print('[INFO] Torch model ready to use !')

if __name__ == '__main__':
	args = parser.parse_args()
	print('[INFO] Importing image')
	if args.input == None :
		img_list = os.listdir(INPUT_DIR)
		img_name = np.random.choice(img_list)
	else :
		img_name = args.input
	image_path = INPUT_DIR + img_name
	print('[INFO] Prepare image for neural network model..')
	image = cv2.imread(image_path)
	image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	im = torch.tensor(np.rollaxis(image, -1,0)/255., dtype=torch.float32)
	print('[INFO] Define checkpoint of intermediate result layer..')
	fhooks, inter_out = [], []
	for i,l in enumerate(list(model.encoder.resnet._modules.keys())):
		fhooks.append(getattr(model.encoder.resnet,l).register_forward_hook(lambda m, input, output: inter_out.append((l,output))))
	print('[INFO] Processing neural network model..')
	encod_out, (capidx, alpha) = model.sample(im.unsqueeze(0), word2idx['<start>'], return_alpha=True)
	capidx, alpha = capidx[0].detach().cpu().numpy(), alpha[0].detach().cpu()
	caption_pred=''.join(list(itertools.takewhile(lambda word: word.strip() != '<end>', map(lambda idx: idx2word[idx]+' ', iter(capidx)))))
	print('[INFO] Processing done !')
	print('[INFO] Results : ' + caption_pred)
	plt.imshow(image); plt.show()
	print('[INFO] Stopping System')