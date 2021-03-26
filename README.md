# Image Classifier: Flowers

This project is a supervised learning problem where we try to predict the category of a flower image using the method of transfer learning. A simple web app is created for the same to upload an image and predict the category.

## Data Collection
1. I used a data set named 17 category flower dataset (https://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
2. The data set has 17 categories of flowers and 80 images for each
3. I used only 3 categories (Colt's Foot, Daisy, Sunflower), thus 240 images

## Data pre-processing
1. Resizing of the image to a particular size
2. Splitting the data into training and test images
3. Initializing the ResNet50 library and adding the preprocessing layer, I used the default 'imagenet' weights

## Model building, training and testing
1. Initializing the model with input of resnet and output as predictions from the model
2. Model compile and optimization 
3. Fitting the model on the training and test sets

## Plotting the output
1. Finally, plotting the Loss and Accuracy (the values are not that great because I did not use a lot of images)
2. Saving the output of the model

## Web app
1. Creating a web app using Flask to deploy the model and have a UI to upload a new image and predict its category, that is, whether the flower is a Colt's Foot or Daisy or Sunflower

Though the accuracy is not great, the app correctly identifies the images moat of the time. My main motivation behind doing this project was to create this web app and see it working, but I am also inspired to collect more data and repeat the experiment to get better results hopefully. 
