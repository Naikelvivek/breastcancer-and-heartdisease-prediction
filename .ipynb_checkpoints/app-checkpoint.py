from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("model.pkl", 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predic():
    featurs = request.form['feature']
    featurs_lst = featurs.split(',')
    np_features = np.asarray(featurs_lst, dtype=np.float32)
    pred = model.predict(np_features.reshape(1, -1))

    output = ["Cancerous" if pred[0] == 1 else "Not Cancerous"]

    return render_template('index.html', message=output)

if __name__ == "__main__":
    app.run()