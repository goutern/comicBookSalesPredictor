# Comic Book Sales Predictor
By: Nicholas Goutermout

# Introduction
This is a project that attempts to use VGGnet, CNN's, Deep Multi layer Perception Networks and ORB feature extraction to predict the sales of a comic book solely based off of its cover. 

# Quick Start
After ensuring you have the correct python dependencies.
Run the scripts in this order
	1. webscraper.py
	2. datacleaner.py
	3. vggFeatureExtractor.py
	4. (optional) orbFeatureExtractor.py
	5. cnn.py
	6. (optional) vggNN.py
	7. (optional) orbNN.py (requires orbFeatureExtractor.py to be run)


## Scripts

The software is divided into three parts, data, feature extraction and NN's. 

The data which is the comic book covers which are located on a google drive at "" or you can use the webscraper.py to fetch the most recent copies and then datacleaner.py to prepare the data. 

The feature extraction section which consists of vggFeatureExtractor.py and orbFeatureExtractor. You must always run the vggFeatureExtractor at least once as this script also prepares the Y data. You only need to run the orbFeatureExtractor if you wish to use the orb features.

Once the data is obtained cleaned and the features extracted you can run any of the three scripts depending on your need or desire. For a Convolution Neural Netowork use cnn.py, for VGG use vggNN.py and for Orb us orbNN.py

## Structure

The project uses a flat structure with the exception that the data set is in a folder called comics in the root of the project. If you downloaded the data from the google drive just extract the folder in the root directory. If you used the web scraped it should already be setup correctly. 

## Feature Extraction

datacleaner.py will make the root numpy array that contains the comic information, cover location and sales data. This file is saved as cleanData.npy

vggFeatureExtractor.py will create four different files. data_y.npy  which consists of the sales targets. image_data.npy which contains the raw image data after it has been reduces to a numpy array and processed by OpenCV.  features.npy which contains the extracted VGG features. features_flatten.npy which contains a flattened array of the feature data. 

orbFeatureExtractor.py will create 10 files features_orb_X.npy and will concatenate them to a single files features_orb.npy. features_orb.npy is then used in the orbNN.

## Predictors

Depending on which method you would like to use to predict the sales you may run cnn.py, vggNN.py, orbNN.py.

cnn.py depends on vggFeatureExtractor.py to be ran on the clean data set.

vggNN.py  depends on vggFeatureExtractor.py to be ran on the clean data set.

orbNN.py depends on depends on vggFeatureExtractor.py and orbFeatureExtractor.py to be run on the clean data set. 

## Python Dependencies

matplotlib
numpy
tensorflow 
keras
scipy
requests  
pandas
html5lib  
bs4
tabulate
time  
cv2
seaborn
sklearn

