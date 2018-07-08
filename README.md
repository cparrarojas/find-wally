**Update 2018.07.08**: now working with the latest version of RetinaNet.

# Find Wally

A *Where's Wally?* solver inspired by [this post](https://towardsdatascience.com/how-to-find-wally-neural-network-eddbb20b0b90) by [Tadej Magajna](https://github.com/tadejmagajna), implemented using [Keras RetinaNet](https://github.com/fizyr/keras-retinanet) with ResNet50 as a backbone, pre-trained on the COCO dataset.

The script is based on Tadej's [`find_wally_pretty`](https://github.com/tadejmagajna/HereIsWally/blob/master/find_wally_pretty.py), plus the [example notebook](https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb) from Keras RetinaNet. I introduced changes to allow for multiple detections in a single image (as in Wally+Wenda below); this is due to the fact that the model is very sensitive to Wally's size, so I included all of his larger appearances&mdash;not part of the puzzles themselves&mdash;in the training annotations.

![Wally + Wenda](https://raw.githubusercontent.com/cparrarojas/find-wally/master/results/wallywenda.png)

## Usage
```
python find_wally.py PATH_TO_MODEL PATH_TO_IMAGE_1 PATH_TO_IMAGE_2...
```

Images are shown one at a time. If no image path is provided, the script will randomly choose one image from the training+validation directory and another one from the testing directory.

The model file can be downloaded from [here](https://github.com/cparrarojas/find-wally/releases/download/0.2/weights.h5).

**Notes:**
- Detections take ~1min per image on CPU.

## Comments on training

Following the instructions from the RetinaNet repository, I trained the model with the default 50 epochs and 1500 steps per epoch, freezing the backbone layers, augmenting the data with random transformations and using an image size larger than default to allow for the proper anchoring of the annotations. In other words, the exact command I ran was:

```bash
$ keras_retinanet/bin/train.py --weights PATH_TO_PRETRAINED_MODEL --steps 1500 --freeze-backbone --random-transform --image-min-side 1800 --image-max-side 3000 csv PATH_TO_annotations.csv PATH_TO_classes.csv --val-annotations PATH_TO_val_annotations.csv
```

On a GeForce GTX1070, training took ~30min per epoch.

## Requirements
- Keras 2.2.0+
- [Keras RetinaNet](https://github.com/fizyr/keras-retinanet)
- Tested using tensorflow 1.4 and Python 3.6

## Sources
- [Training images](https://github.com/vc1492a/Hey-Waldo)
- [Testing images](https://www.flickr.com/photos/153621475@N06/sets/72157684946674930)
- [Pre-trained model](https://github.com/fizyr/keras-retinanet/releases/download/0.3.1/resnet50_coco_best_v2.1.0.h5)
