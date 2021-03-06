# Animate step of understanding image

See the notebook [here](../notebook_computer-vision_ann.ipynb) (Work in progress)

Self-supervised detection - This model is based on DINO [paper](https://arxiv.org/pdf/2104.14294.pdf).

```bash
"A Vision Transformer detect a B-Boy doing a flare"

How does an artificial neural network understand an image? This self-supervised program is the result of a learning process that automatically detect an object on an image with semantic segmentation and where each step of the process is represented in this animation. More details in my github notebook.

#deeplearning #coder #artificialintelligence #computerscience #machinelearning #ai #neuralnetwork #creativecoding #breakdance #tech #dino #facebook #attention #pytorch #opencv
```


###### Attribution required : Fabien Furfaro (CC 4.0 BY NC ND SA)

Convert image sequence with transparency to MP4
```
ffmpeg -i simple_out_%d.png -vcodec png simple_out.mp4
```


## Detection with dino

```
python3 visualize_attention.py --image_path input/dog_bike_car.jpg
# in python3
run visualize_attention.py --image_path input/dog_bike_car.jpg 
```


att = np.concatenate([a[None] for a in attentions], axis = 0)
plt.imshow(np.sum(att,axis=0)); plt.show()



# video and dino visualization :
val /= torch.sum(val, dim=1, keepdim=True)
val, idx = torch.sort(attentions)


testset for segmentation with dino (https://github.com/facebookresearch/dino#evaluation-davis-2017-video-object-segmentation)

https://github.com/davisvideochallenge/davis-2017

https://data.vision.ee.ethz.ch/jpont/davis/DAVIS-2017-test-dev-480p.zip