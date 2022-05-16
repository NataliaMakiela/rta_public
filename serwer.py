import pickle
from flask import Flask, request
from flask_restful import Resource, Api
from flask import jsonify
import numpy as np
from perceptron_model import Perceptron

app = Flask(__name__)

@app.route("/api/predict", methods = ['GET'])
def prediction():

    with open(r"C:\Users\nk131\Desktop\SGH\rta\rta_public\model_rta.pkl", "rb") as fh:    
        loaded_model = pickle.load(fh)

    sl = float(request.args.get("sl"))
    pl = float(request.args.get("pl"))
    p = loaded_model.predict(np.array([sl, pl]))
    
    if p == 1:
        name =  "versicolor"
    else:
        name = "setosa"
        
    return f"Predicted class for sepal_length = {sl}, and petal_length = {pl} is {name}."


app.run(port=5000) 