from keras.applications.mobilenet import preprocess_input
import scipy.misc
import os
from scipy import ndimage
from keras.models import load_model
from keras.preprocessing.image import img_to_array, array_to_img
from keras.applications.mobilenet import decode_predictions, preprocess_input
from PIL import Image
from imagehash import phash
import numpy as np

IMAGE_DIMS = (224, 224)
TREE_FROG_IDX = 31
#TREE_FROG_STR = "malinois"
#FILENAME = "./frog"
TREE_FROG_STR = "tree_frog"
FILENAME = "./trixi"

# I'm pretty sure I borrowed this function from somewhere, but cannot remember
# the source to cite them properly.
def hash_hamming_distance(h1, h2):
    s1 = str(h1)
    s2 = str(h2)
    return sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(s1, s2)))

def is_similar_img(path1, path2):
    image1 = Image.open(path1)
    image2 = Image.open(path2)
    dist = hash_hamming_distance(phash(image1), phash(image2))
    return dist <= 1


def prepare_image(image, target=IMAGE_DIMS):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def get_predictions(image, model):
    preds = model.predict(image)
    dec_preds = decode_predictions(preds, 1000)[0]
    _, label1, conf1 = decode_predictions(preds)[0][0]
    return dec_preds


def create_img(img_path, img_res_path, model_path, target_str, target_idx, des_conf=0.95):
    testTemp = Image.open(img_path).resize(IMAGE_DIMS)
    test = prepare_image(testTemp)
    testTemp.close()
    model = load_model(model_path)
    solved = False
    while solved == False:
        #ADD YOUR CODE FOR AUGMENTING IMAGES IN HERE. Some functions to get you started
        #Note: This code will cause an assertation error because it doesnt create a valid solution.
        print(get_predictions(test, model)[0][1])
        print(get_predictions(test, model)[0][2])
        solved = True

if __name__ == "__main__":
    create_img(FILENAME+".png", FILENAME+"_frog.png", "./model.h5", TREE_FROG_STR, TREE_FROG_IDX)
    assert is_similar_img(FILENAME+".png", FILENAME+"_frog.png")
