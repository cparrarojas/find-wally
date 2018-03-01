# Find Wally

A *Where's Wally?* solver inspired by [this post](https://towardsdatascience.com/how-to-find-wally-neural-network-eddbb20b0b90) by [Tadej Magajna](https://github.com/tadejmagajna), implemented using [Keras RetinaNet](https://github.com/fizyr/keras-retinanet) with ResNet50 as a backbone, pre-trained on the COCO dataset.

The script is based on Tadej's [`find_wally_pretty`](https://github.com/tadejmagajna/HereIsWally/blob/master/find_wally_pretty.py), plus the [example notebook](https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb) from Keras RetinaNet. I introduced changes to allow for multiple detections in a single image; this is due to the fact that the model is very sensitive to Wally's size, so I included all of his larger appearances&mdash;not part of the puzzles themselves&mdash;in the training annotations.

![Wally + Wenda](https://raw.githubusercontent.com/cparrarojas/find-wally/master/results/wallywenda.png)

## Usage
```
python find_wally.py PATH_TO_IMAGE_1 PATH_TO_IMAGE_2...
```

Images are shown one at a time. If no image path is provided, the script will randomly choose one image from training+validation and another one from testing.

**Notes:**
- Detections take ~1min per image on CPU.
- It overfits. A lot. I'll work on that...

## Training (optional)

1. Copy the files from the `keras_retinanet` directory in this repository into your own. This freezes the ResNet50 layers&mdash;the original implementation allows all backbone layers to be retrained&mdash;and modifies the pre-processing parameters to work with larger images and allow proper anchoring of the annotations&mdash;otherwise, most images will not contribute to training.
2. From the `keras-retinanet` repository, run:
```
keras_retinanet/bin/train.py --weights PATH_TO_PRETRAINED_MODEL --steps STEPS_PER_EPOCH csv PATH_TO_annotations.csv PATH_TO_classes.csv --val-annotations PATH_TO_val_annotations.csv
```

I used the default 50 epochs with 1500 steps per epoch. On a GeForce GTX1070, training took ~30min per epoch.

## Requirements
- Keras 2.1.3+
- [Keras RetinaNet](https://github.com/fizyr/keras-retinanet)
- Tested using tensorflow 1.4 and Python 3.6

## Sources
- [Training images](https://github.com/vc1492a/Hey-Waldo)
- [Testing images](https://www.flickr.com/photos/153621475@N06/sets/72157684946674930)
- [Pre-trained model](https://github.com/fizyr/keras-retinanet/releases/download/0.1/resnet50_coco_best_v1.2.2.h5)
