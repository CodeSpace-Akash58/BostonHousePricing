
import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
lin_model=pickle.load(open('lin_model.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

# API use for POSTMAN app
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    result=lin_model.predict(new_data)
    print(result[0])
    return jsonify(result[0])

# User input webpage
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    result=lin_model.predict(final_input)[0]
    print(result)
    return render_template("home.html",prediction_text="The House price prediction is {}".format(result))


if __name__=="__main__":
    app.run(debug=True)
   
