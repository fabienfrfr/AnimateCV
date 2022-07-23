# Animate step of understanding image

See the notebook [here](/notebook_computer-vision_ann.ipynb) (Work in progress)

Image Captionning with Transformer - This model is based on this [paper](https://arxiv.org/abs/1502.03044).

###### Attribution required : Fabien Furfaro (CC 4.0 BY NC ND SA)


```bash
"A person is standing on top of a cliff"

How does an artificial neural network understand an image? This program is the result of a learning process that automatically annotates images caption and where each step of the process is represented in this animation. More details in my github notebook.

#deeplearning #coder #artificialintelligence #computerscience #machinelearning #ai #neuralnetwork #convolution #tech #attention #caption #pytorch #opencv
```


## Tips :

If you have a old nvidia GPU doesn't compatible with CUDA 10.2 and you want to use GPU calculation (like me), you need to do some requirement :

0 - OS compatibilities :
```bash
	- Nvidia doesnt maintain old version of CUDA in new Ubuntu version # for exemple use ubuntu 18.04 LTS for 418-server
```

Other, verify in https://developer.nvidia.com/cuda-gpus the compute capability of your gpu,  the minimum cuda capability that pytorch support is 3.5 (CUDA capability 3.0 support was dropped in v0.3.1). Otherwise, you install pytorch with source (whl file : https://pytorch.org/get-started/previous-versions/), minimum is 0.3.0, for exemple, is "cu75/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl", if your python is 3.6. Download, and run command (order is important) :

	- python3.5 -m pip install torchvision==0.2.1 # adapted for cuda9
	- python3.5 -m pip uninstall torch
	- pip install torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl

But.. you need to add many function changement (name, etc.) if you want to use my code, like that (for exemple, the very basic function "tensor", is "Tensor" in very old version) :

```bash
def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator
```
And then use it like this:
```bash
@rename('new name')
def f():
    pass
print f.__name__
```
Creation of "utils_oldpytorch" in progress.. not recommended now ! Or install python specific version for test ! (a different version of native, if 3.6, choose 3.5 for exemple)

1 - Install the highest version driver :
```bash
	- sudo add-apt-repository ppa:graphics-drivers/ppa # update after
	- sudo ubuntu-drivers autoinstall # before : apt-get remove --purge nvidia-*
	- ubuntu-drivers devices # see the last version
	- sudo apt install nvidia-driver-XXX-XXX # for me : 418-server
	- sudo apt install nvidia-utils-XXX-XXX # idem
	#- sudo apt-get install nvidia-modprobe
```
2 - Intall toolkit (following ubuntu version compatibilities) :
```bash
	- sudo apt install nvidia-cuda-toolkit gcc-6
	(OR)
	- wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb # ubuntu 18.04 but compatible in 20.04
	- sudo apt list --installed | grep cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39
	(OR)
	- wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.168_418.67_linux.run
```

3 - Testing :
```bash
	- nvidia-smi # if doesn't works, reinstall driver and doesn't install toolkit
	- nvcc --version # if toolkit it's installed
```

Cuda 9.1 & drivers 390 in Ubuntu 18.04 (Python 3.6):

	- sudo apt-get install libjpeg-dev zlib1g-dev
	- sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy python-six python3-six build-essential python-pip python3-pip python-virtualenv swig python-wheel python3-wheel libcurl3-dev libcupti-dev

	- https://developer.nvidia.com/cudnn (need to register to install cuDNN)
		- libcudnn7 (deb file download)

	- pip3 install pillow==6.2.0 scikit-build cmake
	- pip3 install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/cu90/torch_stable.html
	- (in Python3) torch.cuda.is_available()

4 - Install adapted python version (for old pytorch, it's python 3.8 max!) :
```bash
	- sudo apt install software-properties-common -y
	- sudo add-apt-repository ppa:deadsnakes/ppa -y
	- sudo apt install python3.8 -y
	- sudo apt install python3.8-dev python3.8-venv python3.8-distutils python3.8-lib2to3 python3.8-gdbm python3.8-tk -y
		- sudo apt install python3.5-dev python3.5-venv python3.5-gdbm python3.5-tk -y
```
5 - Install python package in 3.8 specific version :
```bash
	- python3.8 -m pip install PACKAGES==VERSION
	- python3.8 -m pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html --use-deprecated=html5lib
	- python3.8 -m pip install ipython
	- python3.8 -m pip install matplotlib --force-reinstall
	- python3.8 -m pip install opencv-python
```

6 - Use specific python version :
```bash
	- python3.8 files.py --arg
	(OR)
	- python3.8 -m IPython
		- run files.py --arg
```


ELSE (exemple) :
```bash
	- specify calculation per CPU in pytorch : torch.load(PATH, map_location=torch.device('cpu'))
	- OR generalize : torch.load(PATH, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	- doesnt use command like "tensor.gpu()" directly ! use conditional statement
```