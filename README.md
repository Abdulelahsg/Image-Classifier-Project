# Image-Classifier-Project
### Install 
This project requires the following python libraries to be installed:<br />

*	NumPy 
*	Pandas 
* tensorflow 
* json
* matplotlip

I will train an image classifier to recognize different species of flowers.
Imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. 
 I will be using this dataset from Oxford of 102 flower categories.


### Loading Data
I will use tensorflow_datasets to load the Oxford Flowers 102 dataset. 
This dataset has 3 splits: `train`, `test`, and `validation`.

The validation and testing sets are <b>used to measure the model's performance on data it hasn't seen yet</b>, 
but you'll still need to normalize and resize the images to the appropriate size.

