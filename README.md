# Bird Classification
Bird classification project for CSE 455 Kaggle competition

## Problem Overview
The problem was to train a multiclass image classifier to identify different types of birds and see how it performs on unseen test data. The evaluation metric is the accuracy on the test data.

### Previous work
My code was based on [tutorial 3](https://colab.research.google.com/drive/1EBz4feoaUvz-o_yeMI27LEQBkvrXNc_4?usp=sharing) and [tutorial 4](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing) from class. I also imported the [ResNet-18 model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) through PyTorch as a pretrained model. According to the [paper](https://arxiv.org/abs/1512.03385), the model uses deep residual learning to improve training, and achieved 3.57% error on the ImageNet dataset. It made sense to use this large model with proven results for pretraining.

### Datasets
"The data is images. Of birds." The train and test data were provided in the Kaggle competition. There are 555 names of birds, or classes. For each bird name, there is a directory of training images for it, so we can give the model numerous examples for each class it should identify. 

## My approach, and results
My approach was to first get a large pretrained model (ResNet-18), then finetune it with the training data for this task. I tried different approaches for improving the finetuning process, including image augmentations, resizing training images, and hyperparameter tuning, and found they could make a big (and sometimes surprising) difference. I used ideas I had learned from class along with other approaches I read about.

### Resizing the images
I had to preprocess the data before training and prediction, and this included resizing the images. In the `get_bird_data` function, I resize the images to the same size. I tried sizes of 64x64, 128x128, 256x256, and 512x512. 

### Image augmentations
I applied different additional image augmentations in `get_bird_data`, combining them along with the image resizing using `transforms.Compose`.

### Hyperparameter tuning
#### Learning rate
Momentum
Decay
#### More epochs

## Additional discussion
