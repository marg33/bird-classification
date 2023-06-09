# Bird Classification
Bird classification project for CSE 455 Kaggle competition

## Problem Overview
The problem was to train a multiclass image classifier to identify different types of birds and see how it performs on unseen test data. The evaluation metric is the accuracy on the test data.

### Previous work
My code was based on [tutorial 3](https://colab.research.google.com/drive/1EBz4feoaUvz-o_yeMI27LEQBkvrXNc_4?usp=sharing) and [tutorial 4](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing) from class. I also imported the [ResNet-18 model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) through PyTorch as a pretrained model. According to the [paper](https://arxiv.org/abs/1512.03385), the model uses deep residual learning to improve training, and achieved 3.57% error on the ImageNet dataset. It made sense to use this large model with proven results for pretraining.

Pretraining is really powerful: ResNet-18's default PyTorch weights are from training on ImageNet, which is a dataset much broader than birds, and likely includes none or few of the same pictures. ImageNet also doesn't have the finegrained labels for 555 species of birds like this dataset does. However, the deep training on ImageNet led to weights that are in a really good spot already for image classification, and serve as great preinitialized weights before finetuning on the more specific task.

### Datasets
"The data is images. Of birds." The train and test data were provided in the Kaggle competition. There are 555 names of birds, or classes. For each bird name, there is a directory of training images for it, so we can give the model numerous examples for each class it should identify. 

## My approach, and results
My approach was to first get a large pretrained model (ResNet-18), then finetune it with the training data for this task. I tried different approaches for improving the finetuning process, including image augmentations, resizing training images, and hyperparameter tuning, and found they could make a big (and sometimes surprising) difference. I used ideas I had learned from class along with other approaches I read about.

### Resizing the images
I had to preprocess the data before training and prediction, and this included resizing the images. In the `get_bird_data` function, I resize the images to the same size. I tried sizes of 128x128, 256x256, and 512x512. Images of size 256x256 led to better performance than 128x128 and 512x512. 

The 128x128 images achieved 0.5135 on the test data, while 256x256 achieved 0.687. Here's what the plots of their losses during training looked like:

![image](https://github.com/marg33/bird-classification/assets/44858702/d42b9fd9-2212-4ac0-b957-57f005cf8e81)

Losses for size 256

![image](https://github.com/marg33/bird-classification/assets/44858702/21146a29-20c7-474a-9b46-21940dce08ce)

A 512x512 images model (with different augmentations, so not directly comparable to the above) achieved 0.6065, while the same model but with 256x256 images achieved 0.6395. This surprised me at first, and I looked up reasons why this would happen. I found that images upscaled too far could cause issues as well: "When small images are upscaled and padded with zero, then NN has to learn that the padded portion has no impact on classification." (https://medium.com/analytics-vidhya/how-to-pick-the-optimal-image-size-for-training-convolution-neural-network-65702b880f05)

There are definitely important tradeoffs here -- images that are too big take exponentially longer to process with convolutional neural networks, and may be scaling some up so much that they end up having too much padding, but images too small may not have enough data for the model to extract good features. 256x256 was a good balance.

### Image augmentations
I applied different additional image augmentations in `get_bird_data`, combining them (with the image resizing) using `transforms.Compose`. 

I used size 256x256 for all of these, after having decided a good size from the above tests.

| Augmentations      | Test accuracy |
| ----------- | ----------- |
| RandomCrop, RandomHorizontalFlip      | 0.6395       |
| RandomCrop, RandomHorizontalFlip, RandomVerticalFlip      | 0.6055       |
| RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective      | 0.5345       |
| RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective      | 0.5345       |
| RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomPerspective, RandomAutocontrast, ColorJitter      | 0.6395     |

Image augmentation can be a powerful tool to prevent overfitting, but too much of it can also corrupt the images so much that the model can't learn the right things from them anymore. With my tests, the best choice ended up being just a small number of augmentations that didn't change the images too much. I would like to try more augmentations that are less extreme transformations in the future, however. Some of the augmentations I tried may have changed the images too much, like RandomPerspective and RandomAutocontrast. I could also try reducing the range for those augmentations.

### Hyperparameter tuning

Previous examples above all trained for 5 epochs with learning rate .1, momentum .9, and decay .0005 (default values for momentum and decay). But selecting good hyperparameters is important. Training with good hyperparameters for different epochs also matters -- what learning rate should you use in the beginning vs. near the end? I played around with the number of epochs, and the learning rate, momentum, and decay, changing them for different stages of epochs.

These were all using size 256x256 images and the RandomCrop, RandomHorizontalFlip, and RandomVerticalFlip augmentations as determined to be the best from the previous section.

| Learning rate | Momentum | Decay      | Test accuracy |
| ----------- | ----------- | --- | --- |
| .1 to epoch 5, .01 to epoch 8, .001 to epoch 10, .0001 to epoch 12 | .9 | .0005      | 0.7865       |
| .1 to epoch 5, .01 to epoch 8, .001 to epoch 10, .0001 to epoch 12 | .9 to epoch 5, .75 to epoch 12, | .0005       | 0.79       |
| .1 to epoch 5, .01 to epoch 8, .001 to epoch 10, .0001 to epoch 12 | .9 to epoch 5, .5 to epoch 10, .1 to epoch 12 | .0005     | 0.7875       |
| .1 to epoch 5, .01 to epoch 8, .001 to epoch 10, .0001 to epoch 12 | .9 to epoch 5, .5 to epoch 10, .1 to epoch 12 | 0    | 0.7925       |
| .1 to epoch 5, .01 to epoch 8, .001 to epoch 10, .0001 to epoch 16 | .9 to epoch 5, .5 to epoch 10, .1 to epoch 16 | 0    | 0.794       |

Here is a plot of the losses for the final model in the table, with the best accuracy 0.794:
![image](https://github.com/marg33/bird-classification/assets/44858702/9c55fa84-05a4-44a2-b23f-2549e143e818)

Here is one for the second to last model in the table, with the second best accuracy 0.7925:
![image](https://github.com/marg33/bird-classification/assets/44858702/83294138-6ee4-42b3-ace8-a76f70e07a89)

I ran a bunch of different tests to get to these results


## Additional discussion

### Takeaways



