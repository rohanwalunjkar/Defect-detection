from flask import Flask, request , render_template
from keras.utils import img_to_array
from defect_detection.config import STATIC_FOLDER_PATH , IMAGE_SIZE, RESIZE_FACTOR
from defect_detection.utils.prediction import get_predictions
from defect_detection.utils import helpers
from werkzeug.utils import secure_filename
from PIL import Image
import os

app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def show_prediction():

    print("show_prediction() method running....")
    

    if request.method == 'POST':
        product_type = str(request.form['product'])
        

        #image = request.files['img']
        image = None
        
    
        if image != None:
            image = request.files['img']
            image.save(f"{STATIC_FOLDER_PATH}/input_img.png")

            img = Image.open(image)
        else :
            img_name = request.form['img_type']
            img = Image.open(f"{STATIC_FOLDER_PATH}/{img_name}")
            img.save(f"{STATIC_FOLDER_PATH}/input_img.png")

            

        # img_size = int(request.form['img_size'])
        # IMAGE_SIZE = (img_size, img_size)
        
        # img = img.resize(IMAGE_SIZE)
        
    
        probs, pt1, pt2  = get_predictions(img, product_type)
        return "<p>success</p>"

        """result = "Defective" if probs[0][0] > probs[0][1] else "Non defective"

        if result=='Defective' :
            img_name = helpers.draw_rectangle(img.resize(size=(8*RESIZE_FACTOR,8*RESIZE_FACTOR)), pt1, pt2)
            return render_template('result.html', probs=probs, img_name = img_name, result=result)
        else :
            return render_template('result.html', probs=probs, result=result)"""
        
        

@app.route("/test")
def test():
    return "<p>Working fine. </p>"





if __name__ == '__main__':
    app.run(debug=False)