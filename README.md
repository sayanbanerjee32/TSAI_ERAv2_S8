# TSAI_ERAv2_S8

## Objective
1. use dataset to CIFAR10
2. Make this network:
  - C1 C2 __c3__ __P1__ C4 C5 C6 __c7 P2__ C8 C9 C10 GAP __c11__
  - CN is convolution layer
  - __cN__ is 1x1 Layer
  - __PN__ is max pooling layer
  - Keep the parameter count less than 50000
  - Max Epochs is 20
3. You are making 3 versions of the above code (in each case achieve above 70% accuracy):
  - Network with Group Normalization
  - Network with Layer Normalization
  - Network with Batch Normalization

## Dataset - CIFAR10

1. Training data size -50,000
2. Image shape - 3 x 32 x 32 (colour image - RGB)
3. Number of classes 10 - plane, car, bird, cat, deer, dog, frog, horse, ship, truck

Example images from training set  
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S8/assets/11560595/711aed42-d235-45f3-b7e1-729fbb8a01fe)



## Model with Batch Normalization

Model architecture is exactly same as the one mentioned in the objective section. The model has 36,272 parameters and receptive field of 44.

### Model Performance

1. Training accuracy - 68.64%
2. Test accuracy - 71.62%

### Training and testing loss and accuracy comparison for each epoch
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S8/assets/11560595/0131ad55-a2b2-4953-bb80-88b62513f286)

### Error Analysis
There are total 2838 wrong classifications out of 10,000 test images. Top 5 confused cases of prediction
|target|prediction|number of images|
|------|----------|----------------|
|deer|bird|156|
|dog|bird|156|
|dog|cat|147|
|cat|bird|146|
|cat|dog|144|


Example of 10 misclassified images in descending order of loss. 
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S8/assets/11560595/e227cb7a-8c70-493a-9e91-06716ddb3068)

### Findings for normalization technique

Batch Normalisation is the most effective normalization technique for this model. The model reached 70% accuracy easily in 20 epochs and even with comparatively higher dropout (0.1) and comparatively less parameters. However, the line for test loss / accuracy is very choppy.

## Model with Layer Normalization

Model architecture is exactly same as the one mentioned in the objective section. The model has 44,400 parameters and receptive field of 44.

### Model Performance

1. Training accuracy - 70%
2. Test accuracy - 71.85%

### Training and testing loss and accuracy comparison for each epoch
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S8/assets/11560595/22fc57f9-352c-41d5-a660-08504cc23182)

### Error Analysis
There are total 2815 wrong classifications out of 10,000 test images. Top 5 confused cases of prediction
|target|prediction|number of images|
|------|----------|----------------|
|cat|dog|230|
|dog|cat|127|
|deer|horse|108|
|bird|dog|98|
|frog|cat|85|
  
Example of 10 misclassified images in descending order of loss. 
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S8/assets/11560595/db450619-bc4f-44e2-8fc5-9f973baaf797)


### Findings for normalization technique

The model reached 70% accuracy in 20 epochs and with comparatively lower dropout (0.05) and comparatively more parameters. This indicates that the model found it difficult to converge on the loss due to the normalization technique as everything else was exactly same compared to batch normalisation model. However, the line for test loss / accuracy is comparatively smooth.

## Model with Group Normalization

Model architecture is exactly same as the one mentioned in the objective section. The model has 36,272 parameters and receptive field of 44.

### Model Performance

1. Training accuracy - 69.27%
2. Test accuracy - 70.77%

### Training and testing loss and accuracy comparison for each epoch
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S8/assets/11560595/56681db9-c0eb-414d-82af-5003e0cb7ce0)

### Error Analysis
There are total 2923 wrong classifications out of 10,000 test images. Top 5 confused cases of prediction
|target|prediction|number of images|
|------|----------|----------------|
|cat|dog|185|
|dog|cat|153|
|deer|horse|139|
|deer|bird|127|
|plane|ship|103|

Example of 10 misclassified images in descending order of loss. 
![image](https://github.com/sayanbanerjee32/TSAI_ERAv2_S8/assets/11560595/74f98fcf-bf31-418d-9029-89c16911b06f)

### Findings for normalization technique

The model reached 70% accuracy in 20 epochs and with comparatively lower dropout (0.05) but same number of parameters compared to batch normalization model. This indicates that the model found it difficult to converge on the loss due to the normalization technique as everything else was exactly same compared to batch normalisation model. And the line for test loss / accuracy is very choppy.
