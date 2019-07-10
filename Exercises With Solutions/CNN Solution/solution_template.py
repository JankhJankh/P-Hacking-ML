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
#FILENAME = "./froggo"
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
    gtconf = 0.0
    loc = 1000
    while solved == False:
        testTemp = Image.open(FILENAME + '_mod.png').resize(IMAGE_DIMS)
        test = prepare_image(testTemp)
        testTemp.close()
        if(get_predictions(test, model)[0][1] == TREE_FROG_STR and get_predictions(test, model)[0][2] > 0.95):
            solved = True
            image = ndimage.imread(FILENAME+'_mod.png')
            scipy.misc.toimage(image, cmin=0.0, cmax=255).save(FILENAME+'_frog.png')
        else:
            image = ndimage.imread(FILENAME+'_mod.png')
            noise = np.random.normal(0,1,(1920,1080,3))
            imagemod = image + noise
            scipy.misc.toimage(imagemod, cmin=0.0, cmax=255).save(FILENAME+'_temp.png')
            testTemp = Image.open(FILENAME+'_temp.png').resize(IMAGE_DIMS)
            test = prepare_image(testTemp)
            testTemp.close()
            preds = get_predictions(test, model)
            for a in range(0,loc):
                if(float(preds[a][2]) > gtconf and preds[a][1] == TREE_FROG_STR):
                    scipy.misc.toimage(imagemod, cmin=0.0, cmax=255).save(FILENAME+'_mod.png')
                    gtconf = float(get_predictions(test, model)[a][2])
                    print('Location: ', a, '| Confidence: ', get_predictions(test, model)[a][2])
                    loc = a+1

if __name__ == "__main__":
    create_img(FILENAME+".png", FILENAME+"_frog.png", "./model.h5", TREE_FROG_STR, TREE_FROG_IDX)
    assert is_similar_img(FILENAME+".png", FILENAME+"_frog.png")
