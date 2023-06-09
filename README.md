# Bird Classification
Bird classification project for CSE 455 Kaggle competition

## Problem Overview
The problem was to train a multiclass image classifier to identify different types of birds. The evaluation metric is the accuracy on the test data. The code is [here](https://github.com/marg33/bird-classification/blob/main/bird-classifier.ipynb).

### Previous work
My code was based on [tutorial 3](https://colab.research.google.com/drive/1EBz4feoaUvz-o_yeMI27LEQBkvrXNc_4?usp=sharing) and [tutorial 4](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing) from class. I also imported the [ResNet-18 model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) through PyTorch as a pretrained model. According to the [paper](https://arxiv.org/abs/1512.03385), the model uses deep residual learning to improve training, and achieved 3.57% error on the ImageNet dataset. It made sense to use this large model with proven results for pretraining.

Pretraining is really powerful: ResNet-18's default PyTorch weights are from training on ImageNet, which is a dataset much broader than birds, and likely includes none or few of the same pictures. ImageNet also doesn't have the finegrained labels for 555 species of birds like this dataset does. However, the deep training on ImageNet led to weights that are in a really good spot already for image classification, and serve as great preinitialized weights before finetuning on the more specific task.

### Datasets
"The data is images. Of birds." The train and test data were provided in the Kaggle competition. There are 555 names of birds, or classes. For each bird name, there is a directory of training images for it, so we can give the model numerous examples for each class it should identify. 

## My approach, and results
My approach was to first get a large pretrained model (ResNet-18), then finetune it with the training data for this task. The finetuning was done with stochastic gradient descent to optimize cross-entropy loss. I tried different approaches for improving the finetuning process, including image augmentations, resizing training images, and hyperparameter tuning, and found they could make a big (and sometimes surprising) difference. I used ideas I had learned from class along with other approaches I read about.

### Resizing the images
I had to preprocess the data before training and prediction, and this included resizing the images. In the `get_bird_data` function, I resize the images to the same size. I tried sizes of 128x128, 256x256, and 512x512. Images of size 256x256 led to better performance than 128x128 and 512x512. 

The 128x128 images achieved 0.5135 on the test data, while 256x256 achieved 0.687. Here's what the plots of their losses during training looked like:

![image](https://github.com/marg33/bird-classification/assets/44858702/d42b9fd9-2212-4ac0-b957-57f005cf8e81)

Losses for size 256

![image](https://github.com/marg33/bird-classification/assets/44858702/21146a29-20c7-474a-9b46-21940dce08ce)

A 512x512 images model (with different augmentations, so not directly comparable to the above) achieved 0.6065, while the same model but with 256x256 images achieved 0.6395. This surprised me at first, and I looked up reasons why this would happen. I found that images upscaled too far could cause issues as well: "When small images are upscaled and padded with zero, then NN has to learn that the padded portion has no impact on classification." (https://medium.com/analytics-vidhya/how-to-pick-the-optimal-image-size-for-training-convolution-neural-network-65702b880f05)

There are definitely important tradeoffs here -- images that are too big take exponentially longer to process with convolutional neural networks, and may be scaling some up so much that they end up having too much padding, but images too small may not have enough data for the model to extract good features. 256x256 was a good balance.

### Image augmentations
I applied different additional image augmentations in `get_bird_data`, combining them (with the image resizing) using `transforms.Compose`. I tried some unique ones we hadn't talked about it class.

I used size 256x256 for all of these, after having decided a good size from the above tests.

| Augmentations      | Test accuracy |
| ----------- | ----------- |
| RandomCrop, RandomHorizontalFlip      | 0.687       |
| RandomCrop, RandomHorizontalFlip, RandomVerticalFlip      | 0.6055       |
| RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective      | 0.5345       |
| RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective      | 0.5345       |
| RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomPerspective, RandomAutocontrast, ColorJitter      | 0.6395     |

Here's a picture of some train data images with a lot of augmentations applied (from the last row in the table):
![image](https://github.com/marg33/bird-classification/assets/44858702/269b529f-f066-4bdd-92f9-fbe78b6c689e)

Image augmentation can be a powerful tool to prevent overfitting, but too much of it can also corrupt the images so much that the model can't learn the right things from them anymore. With my tests, the best choice ended up being just a small number of augmentations that didn't change the images too much, that is, the first test in the table, which achieved an accuracy of 0.687. The last test in the table, of using a large number of augmentations, had the second highest accuracy of 0.6395, but it was still worse than the simple augmentations. A better selection of augmentations may have beaten it out, though.

Augmentations are still promising, and as seen in the next section, overfitting to training data may have been an issue, and they would help to reduce that. I would like to try more varied augmentations that are less extreme transformations in the future. Some of the augmentations I tried may have changed the images too much, like RandomPerspective. I could also try reducing the range for those augmentations.

### Hyperparameter tuning

Previous examples above all trained for 5 epochs with learning rate .1, momentum .9, and decay .0005 (default values for momentum and decay). But selecting good hyperparameters is important. Training with good hyperparameters for different epochs also matters -- what learning rate should you use in the beginning vs. near the end? I played around with the number of epochs, and the learning rate, momentum, and decay, changing them for different stages of epochs.

These were all using size 256x256 images and the RandomCrop and RandomHorizontalFlip augmentations as determined to be the best from the previous section.

| Learning rate | Momentum | Decay      | Test accuracy |
| ----------- | ----------- | --- | --- |
| .1 to epoch 5, .01 to epoch 7 | .9 | .0005      | 0.6865      |
| .1 to epoch 5, .01 to epoch 8, .001 to epoch 10, .0001 to epoch 12 | .9 | .0005      | 0.7865       |
| .1 to epoch 5, .01 to epoch 8, .001 to epoch 10, .0001 to epoch 12 | .9 | .0005      | 0.7865       |
| .1 to epoch 5, .01 to epoch 8, .001 to epoch 10, .0001 to epoch 12 | .9 to epoch 5, .75 to epoch 12, | .0005       | 0.79       |
| .1 to epoch 5, .01 to epoch 8, .001 to epoch 10, .0001 to epoch 12 | .9 to epoch 5, .5 to epoch 10, .1 to epoch 12 | .0005     | 0.7875       |
| .1 to epoch 5, .01 to epoch 8, .001 to epoch 10, .0001 to epoch 12 | .9 to epoch 5, .5 to epoch 10, .1 to epoch 12 | 0    | 0.7925       |
| .1 to epoch 5, .01 to epoch 8, .001 to epoch 10, .0001 to epoch 16 | .9 to epoch 5, .5 to epoch 10, .1 to epoch 16 | 0    | 0.794       |

Here is a plot of the losses for the final model in the table, with the best accuracy 0.794:

![image](https://github.com/marg33/bird-classification/assets/44858702/9c55fa84-05a4-44a2-b23f-2549e143e818)

Here is one for the second to last model in the table, with the second best accuracy 0.7925:

![image](https://github.com/marg33/bird-classification/assets/44858702/83294138-6ee4-42b3-ace8-a76f70e07a89)

#### General strategy

I ran a bunch of different tests to find what hyperparameters I needed. My general strategy was to allow the model to train with certain hyperparameters (learning rate, momentum, and decay) for a number of epochs, and look at the trend of the losses during training. If the losses in an epoch were no longer decreasing, I would try a lower learning rate and/or momentum from that epoch on. It made sense to decrease the learning rate as losses became smaller and smaller and closer to converging.

#### Momentum

I wasn't as sure about how to use momentum. Momentum seemed like it could definitely be promising to adjust, but any changes in momentum can make big differences in training, sometimes for the worse. I noticed the default momentum value in the train function from the tutorials was 0.9. Wondering why that was, I looked up an article discussing it: [Why 0.9? Towards Better Momentum Strategies in Deep Learning.](https://towardsdatascience.com/why-0-9-towards-better-momentum-strategies-in-deep-learning-827408503650).

> Momentum is a widely-used strategy for accelerating the convergence of gradient-based optimization techniques. Momentum was designed to speed up learning in directions of low curvature, without becoming unstable in directions of high curvature. In deep learning, most practitioners set the value of momentum to 0.9 without attempting to further tune this hyperparameter (i.e., this is the default value for momentum in many popular deep learning packages). However, there is no indication that this choice for the value of momentum is universally well-behaved.

The article explores strategies using momentum decay, reducing momentum more and more in later epochs of training. I thought this made sense as a heuristic, since a larger momentum could help overcome difficult curvatures early on when the gradient is very far from converging and it may be ideal to take larger steps, but in later stages, high momentum could hinder converging as you may need to take smaller steps to not overshoot. I tried implementing this by reducing momentum along with learning rate in later epochs of training, and it did help, although not by a huge amount: a model with momentum decay achieved a test accuracy of 0.79 (row 4), while the same model with 0.9 momentum throughout achieved 0.7865 accuracy (row 3).

#### Decay

Decay is another hyperparameter that I didn't know much about before. The default value in the tutorials' train function was 0.0005. I hadn't come across that value in most neural networks I had seen before, so I tried searching around again. I found interesting information in [How to Use Weight Decay to Reduce Overfitting of Neural Network in Keras](https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/), which talked about decay values used in different models. Regarding CNNs, it said:

> Weight regularization does not seem widely used in CNN models, or if it is used, its use is not widely reported.
>
> L2 weight regularization with very small regularization hyperparameters such as (e.g. 0.0005 or 5 x 10^−4) may be a good starting point.

It went on to list some papers using this value for decay.

I kept the default 0.0005 decay value for some of my tests, then tried using a decay of 0. This helped to push accuracy a little higher. For example, the model in row 5 uses the 0.0005 decay value and achieves 0.7875 test accuracy, while the model in row 6, which is the same except it has 0 decay, gets to 0.7925 test accuracy.

## Additional discussion

While hyperparameter changes and longer training led to much better accuracy, of over a 10% test accuracy improvement over the best models that only trained for 5 epochs listed ealier, they fit much more to the training data than test data. This is to be expected with ML models, but finding more strategies to improve generalizability to unseen data would be useful. More regularization techniques and stategies to prevent the model from memorizing image artifacts (such as augmentations) would be good to try, but I would also need to be careful to make sure they aren't so excessive that they prevent the model from learning good features.

I would like to try more momentum, decay, and learning rate changes in the future. It takes a lot of experimenting to figure out what hyperparameters work, since there is almost no way to predict the effects without trying them out. I think there's a lot of room for experimentation to get the accuracy even higher.

One problem was not being able to get the test accuracy of many models since Kaggle limited the number of submissions. This made making changes hard sometimes because I wouldn't be sure which direction I should go in with few accuracies to compare. I could try setting apart some train data for a validation set next time.

My approach of first determining the data preprocessing techniques I wanted for the model (augmentations and resizing), then fiddling with hyperparameters and extra epochs, was useful for going through the parts of the classifier step by step and optimizing each step as much as possible. However, a different strategy of experimenting with all of these variables without being limited to the sequence (for example, changing augmentations first, then training hyperparameters, then augmentations again) could also be helpful to try, to maximize the potential of improvement from all parts.

This project was really fun and satisfying to do -- I loved watching the accuracy get higher and higher with later models, and I learned a lot about neural networks and image classification by implementing this classifier and finding strategies to improve it. Some outcomes were not what I expected at first, and synthesizing what I learned from class and further reading was rewarding.
