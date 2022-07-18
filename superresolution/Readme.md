# Animate step of understanding image

See the notebook [here](../notebook_computer-vision_ann.ipynb) (Work in progress)

Super-resolution with GAN - This model is based on this [paper](https://arxiv.org/abs/1609.04802).

###### Attribution required : Fabien Furfaro (CC 4.0 BY NC ND SA)


Le modèle a été entrainé par **sgrvinod**. Pour etre inclu dans le GitHub, il a été splité par la commande :

```bash
split -b 20M checkpoint_srgan.pth.tar --verbose
```

Pour l'utilisation, soit vous le telechargé à l'adresse https://drive.google.com/file/d/1_PJ1Uimbr0xrPjE8U3Q_bG7XycGgsbVo/, soit vous le concaténé via la commade (implementé en python) :

```bash
cat x* > checkpoint_srgan.pth.tar
```

Il convient ensuite de supprimer les fichiers "x", pour éviter les doublons..