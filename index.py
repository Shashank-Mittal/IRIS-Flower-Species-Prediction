from flask import *
from sklearn.externals import joblib
import numpy as np
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/filtered",methods=["GET"])
def filtered():
    try:
        sepal_length = float(request.args.get('input1'))
        sepal_width = float(request.args.get('input2'))
        petal_length = float(request.args.get('input3'))
        petal_width = float(request.args.get('input4'))
    except:
        return "Please provide values in Centimeters only"
    prediction_data = np.array([sepal_length,sepal_width,petal_length,petal_width], ndmin=2) 
    trained_object = joblib.load('iris.pkl')
    result = trained_object.predict(prediction_data)
    if(result==1):
        return "Iris-versicolor"
    if(result==2):
        return "Iris-virginica"
    return "Iris-setosa"

if __name__ == '__main__':
    app.run(debug=True)
