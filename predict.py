import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from PIL import Image
import argparse
import json

parser = argparse.ArgumentParser(description='Image Classifier - Part(2))')
parser.add_argument('--input', default='./test_images/wild_pansy.jpg', action="store", type = str, help='image path')
parser.add_argument('--model', default='./1604730625.h5', action="store", type = str, help='checkpoint file path/name')
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int, help='return top K most likely classes')
parser.add_argument('--category_names', dest="category_names", action="store", default='label_map.json', help='mapping the categories to real names')


arg_parser = parser.parse_args()

image_path = arg_parser.input
model_path = arg_parser.model
topk = arg_parser.top_k
category_names = arg_parser.category_names


def process_image(image):
    image =  tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image,(224, 224))
    image /=255
    return image.numpy()

def predict(image_path, model, top_k):
    
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    modified_img_dimension = np.expand_dims(image, axis=0)
    
    
    probs = model.predict(modified_img_dimension)
    probabilities, classes = tf.math.top_k(probs, k=top_k)
    
    probabilities = probabilities.numpy()
    classes = classes.numpy()
    
    return probabilities, classes


if __name__== "__main__":

    print ("start Prediction ...")
    print ("topk:", topk)
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    loaded = tf.keras.models.load_model(model_path, custom_objects ={'KerasLayer':hub.KerasLayer}, compile = False)
    probs, classes = predict(image_path, loaded, topk)
    print('Below are the probs and classes')
    print('probs', probs)
    print('classes', classes)

 
   




