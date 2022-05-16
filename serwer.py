import pickle
from flask import Flask, request
from flask_restful import Resource, Api
from flask import jsonify
import numpy as np
from perceptron_model import Perceptron

app = Flask(__name__)

@app.route("/api/predict", methods = ['GET'])
def prediction():

    with open("model_rta.pkl", "rb") as fh:    
        loaded_model = pickle.load(fh)

    sl = float(request.args.get("sl"))
    pl = float(request.args.get("pl"))
    p = loaded_model.predict(np.array([sl, pl]))
    
    if p == 1:
        name =  "versicolor"
    else:
        name = "setosa"
        
    return f"Predicted class for sepal_length = {sl}, and petal_length = {pl} is {name}."


# to build the image: docker build -t <image_name> .
# to run the image: docker run -p 5000:5000 -t -i <image_name>

# http://127.0.0.1:5000/api/predict?&sl=4.5&pl=3.2

#import requests
#response = requests.get("http://127.0.0.1:5000/api/predict?&sl=4.5&pl=3.2")
#print(response.content)