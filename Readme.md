# Remaster old film & recognition

The objective is to remaster old film with deep learning tool. # not now

```
sudo apt install nvidia-cuda-toolkit

!pip3 install opencv-python
!pip3 install dlib
!pip3 install imutils

!pip install scikit-image
!pip install scipy
!pip install Pillow
!pip install pycuda

!ffmpeg -i Movie.webm Movie.mp4

# si modele keras
!pip3 install keras
!pip3 install tensorflow

# si modele pytorch
!pip3 install torch
!pip3 install torchvision
```

Animate attention Algorithm : https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning or https://github.com/Subangkar/Image-Captioning-Attention-PyTorch

```
python3 caption.py -i input/cat.jpg -m model/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar -wm model/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json
# in ipython3
run caption.py -i input/cat.jpg -m model/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar -wm model/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json
# breakpoint for debugging (line 42 in this example, not empty or comment line)
run -d -b42 filename.py
```







**To add :**
The recognition is for Visual impairment or Deafness, the objective is to describe the scenes during the film.


using pre-trained model :
https://towardsdatascience.com/4-pre-trained-cnn-models-to-use-for-computer-vision-with-transfer-learning-885cb1b2dfc

https://www.vlfeat.org/matconvnet/pretrained/

https://www.analyticsvidhya.com/blog/2018/07/top-10-pretrained-models-get-started-deep-learning-part-1-computer-vision/


r-cnn vs YOLO
https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e

Image captionning (titre des video)
https://youtu.be/y2BaTt1fxJU
https://arxiv.org/abs/1502.03044

Database of pretrained model :
https://modelzoo.co/
https://pjreddie.com/darknet/
https://towardsdatascience.com/5-websites-to-download-pre-trained-machine-learning-models-6d136d58f4e7