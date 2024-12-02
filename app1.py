import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the scaler and model from pickle files
sc = pickle.load(open('sc1.pkl', 'rb')) 
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    lst = [] 
    
    # Retrieve input values from the form
    cp = int(request.form['cp'])
    if cp == 0:
        lst += [1, 0, 0, 0]
    elif cp == 1:
        lst += [0, 1, 0, 0]
    elif cp == 2:
        lst += [0, 0, 1, 0]
    elif cp >= 3:
        lst += [0, 0, 0, 1]

    trestbps = int(request.form["trestbps"])
    lst += [trestbps]

    chol = int(request.form["chol"])
    lst += [chol]

    fbs = int(request.form["fbs"])
    if fbs == 0:
        lst += [1, 0]
    elif fbs == 1:
        lst += [0, 1]

    restecg = int(request.form["restecg"])
    if restecg == 0:
        lst += [1, 0, 0]
    elif restecg == 1:
        lst += [0, 1, 0]
    else:
        lst += [0, 0, 1]

    thalach = int(request.form["thalach"])
    lst += [thalach]

    exang = int(request.form["exang"])
    if exang == 0:
        lst += [1, 0]
    else:
        lst += [0, 1]

    # Convert the list to a numpy array and scale it
    final_features = [np.array(lst)]
    prediction = model.predict(sc.transform(final_features))

    # Output either 'Yes' or 'No' based on the prediction
    if prediction == 1:
        result = "Heart Disease Predicted"
    else:
        result = "No Heart Disease Detected"

    return render_template('result.html', message=result)


if __name__ == '__main__':
    app.run(debug=True, port = 5001)
