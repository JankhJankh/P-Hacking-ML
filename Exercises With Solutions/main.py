import os
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn import cluster
import numpy as np
from sklearn.cluster import KMeans
import sys
from PIL import Image
from flask import Flask, render_template, request, send_from_directory
from keras.applications.mobilenet import decode_predictions, preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import io
from imagehash import phash
import warnings
import os

ALLOWED_EXTENSIONS = set(['png'])


warnings.filterwarnings('ignore')
# Probably not needed for production, but I have GPU support enabled on my version
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


TREE_FROG_ID = 31
TREE_FROG_STR = "tree_frog"
THRESHOLD = 0.95
PHASH_TRESH = 2
IMAGE_DIMS = (224, 224)
FLAG = "sectalks_flag{4dv3rs3rial_n015e_>_C0nvN3ts}"
KmeanBadFlag = "sectalks_flag{D4t4_15_b3tt3r_w1th_sup3rv1s1on}"
KmeanPerfFlag = "sectalks_flag{P-H4ck3d_your_w4y_to_100%}"


app = Flask(__name__)
model = None
base_img = None
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
# Heavily taken from https://blog.keras.io/category/tutorials.html

def import_model():
    global model
    model = load_model("./static/cnn/model.h5")
    global base_img
    base_img = Image.open("./static/cnn/img/trixi.png").resize(IMAGE_DIMS)    

def prepare_image(image):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(IMAGE_DIMS)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # return the processed image
    return image


def get_predictions(image):
    preds = model.predict(image)
    dec_preds = decode_predictions(preds)[0]
    _, label1, conf1 = decode_predictions(preds)[0][0]
    return label1, conf1, dec_preds


def hash_hamming_distance(h1, h2):
    s1 = str(h1)
    s2 = str(h2)
    return sum(map(lambda x: 0 if x[0] == x[1] else 1, zip(s1, s2)))


@app.route("/predict", methods=["POST"])
def predict():
    global model

    # Due to some wonkiness with how Flask is loaded and the Keras backend,
    # the model has issues when it's loaded elsewhere.
    if model is None:
        import_model()

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("frog"):

            frog_img = Image.open(io.BytesIO(request.files["frog"].read()))

            frog_dist = hash_hamming_distance(phash(frog_img), phash(base_img))
            frog_mat = prepare_image(frog_img)
            
            # read the image in PIL format
            frog_label, frog_conf, top_preds = get_predictions(frog_mat)
            
            res = {}
            res["is_frog"] = TREE_FROG_STR in frog_label 
            res["frog_conf"] = frog_conf
            res["frog_cat"] = frog_label
            res["frog_img_sim"] = frog_dist
            res["top_5"] = top_preds

            if TREE_FROG_STR in frog_label and frog_conf >= THRESHOLD and frog_dist <= PHASH_TRESH:
                return render_template("win.html", flag=FLAG)
            else:
                return render_template("results.html", results=res)

    return "Image processing fail"


@app.route('/static/cnn/<path:path>')
def send_img(path):
    return send_from_directory('static/cnn/', path)


@app.route("/cnn.html", methods=['GET'])
def index():
    return render_template("cnn.html")

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def main_page():
	return render_template('index.html')

@app.route('/kmeansupload')
def upload_form():
	return render_template('kmeansUpload.html')

@app.route('/kmeansresult')
def kmeans_result():
	return render_template('kmeansResult.html')

@app.route('/kmeans.html')
def kmeans_upload_form():
	path = "static/kmeans"
	files = []
	for r, d, f in os.walk(path):
		for file in f:
			if '.png' in file:
				files.append(os.path.join(r.replace("static/",""), file).replace("\\","/"))
	print(files)
	return render_template('kmeans.html', files = files)

@app.route('/kmeans', methods=['POST'])
def kmeans_clear():
	if request.method == 'POST':
		os.system("ls static/kmeans/*.png")
	return redirect('/kmeans')

@app.route('/runkmeans', methods=['POST'])
def kmeans_run():
	count = 0
	for filename in os.listdir("static/kmeans/"):
		if filename.endswith(".png"):
			count = count + 1				
			image = plt.imread("static/kmeans/" + filename)
			data = image.reshape(1,32*32*3)
			if 'test' in locals():
				test = np.concatenate((test, data))
			else:
				test = data
	
	kmeans = KMeans(n_clusters=3, random_state=0).fit(test)
	class1 = []
	class2 = []
	class3 = []
	scores = [0,0,0,0,0,0,0,0,0]
	for filename in os.listdir("static/kmeans/"):
		if filename.endswith(".png"):
			image = plt.imread("static/kmeans/" + filename)
			data = image.reshape(1,32*32*3)
			clusters = kmeans.predict(data)
			if(clusters[0] == 0):
				if(filename.startswith("dog")):
					scores[0] +=1
				if(filename.startswith("auto")):
					scores[1] +=1
				if(filename.startswith("ship")):
					scores[2] +=1
				class1.append("kmeans/" + filename)
			if(clusters[0] == 1):
				if(filename.startswith("dog")):
					scores[3] +=1
				if(filename.startswith("auto")):
					scores[4] +=1
				if(filename.startswith("ship")):
					scores[5] +=1
				class2.append("kmeans/" + filename)
			if(clusters[0] == 2):
				if(filename.startswith("dog")):
					scores[6] +=1
				if(filename.startswith("auto")):
					scores[7] +=1
				if(filename.startswith("ship")):
					scores[8] +=1
				class3.append("kmeans/" + filename)
	flag = ""
	avg =   (scores[0]+scores[1]+scores[2]+scores[3]+scores[4]+scores[5]+scores[6]+scores[7]+scores[8])/9 - 10
	if((scores[0]+scores[1]+scores[2] > 300) and (scores[0] * scores[1] == 0) and (scores[3]+scores[4]+scores[5] > 300) and (scores[3] * scores[4] == 0) and (scores[6]+scores[7]+scores[8] > 300) and (scores[6] * scores[7] == 0)):
		flag = KmeanPerfFlag
	elif(scores[0] > avg and scores[1] > avg and scores[2] > avg and scores[3] > avg and scores[4] > avg and scores[5] > avg and scores[6] > avg and scores[7] > avg and scores[8] > avg):
		flag = KmeanBadFlag
	return render_template('kmeansResult.html', class1 = class1, class2 = class2, class3 = class3, scores = scores, flag = flag)

@app.route('/kmeansupload', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		images = request.files.to_dict() #convert multidict to dict
		for image in images:     #image will be the key
			file_name = secure_filename(images[image].filename)
			images[image].save(os.path.join('static/kmeans/', file_name))
			return redirect('/kmeans')

@app.after_request
def set_response_headers(response):
	response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
	response.headers['Pragma'] = 'no-cache'
	response.headers['Expires'] = '0'
	return response

if __name__ == "__main__":
    app.run()
