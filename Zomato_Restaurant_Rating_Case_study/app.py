import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
dtmodel=pickle.load(open('DecisionTreeModel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    data = {
        'Votes': data['Votes'],
        'Average Cost for two':data['Average Cost for two'],
	    'Has Table booking':data['Has Table booking'],
	    'Has Online delivery':data['Has Online delivery'],
	    'Price range':data['Price range']
    }

    if data['Has Table booking'] == 'Yes':
        data['Has Table booking'] = 1
    elif data['Has Table booking'] == 'No':
        data['Has Table booking'] = 0

    if data['Has Online delivery'] == 'Yes':
        data['Has Online delivery'] = 1
    elif data['Has Online delivery'] == 'No':
        data['Has Online delivery'] = 0
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=dtmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data =request.form.to_dict()
    data = {
        'Votes': data['Votes'],
        'Average Cost for two':data['Average Cost for two'],
	    'Has Table booking':data['Has Table booking'],
	    'Has Online delivery':data['Has Online delivery'],
	    'Price range':data['Price range']
    }
    if data['Has Table booking'] == 'Yes':
        data['Has Table booking'] = 1
    elif data['Has Table booking'] == 'No':
        data['Has Table booking'] = 0

    if data['Has Online delivery'] == 'Yes':
        data['Has Online delivery'] = 1
    elif data['Has Online delivery'] == 'No':
        data['Has Online delivery'] = 0

    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=dtmodel.predict(new_data)
    print(output[0])
    return render_template("home.html",prediction_text="Zomato Restaurant Rating predicion is {}".format(np.round(output[0]),1))



if __name__=="__main__":
    app.run(debug=True)
   
     