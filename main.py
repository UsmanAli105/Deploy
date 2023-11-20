from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
import configparser
import numpy as np
import pandas as pd
import sys
import logging
import traceback
import joblib
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

app = Flask(__name__)

# Read configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

# Configure BasicAuth
app.config['BASIC_AUTH_USERNAME'] = config['Credentials']['USERNAME']
app.config['BASIC_AUTH_PASSWORD'] = config['Credentials']['PASSWORD']
basic_auth = BasicAuth(app)

@app.route('/predict', methods=['POST'])
@basic_auth.required
def predict():
    if model:
        try:
            json_ = request.json
            df_row = pd.DataFrame.from_dict([json_])
            prediction = list(model.predict(df_row))
            label = 'Attack' if prediction[0] else 'Normal'
            return jsonify({
                'prediction': str(prediction),
                'Label': label
            })
        except ValueError as ve:
            return jsonify({'trace': traceback.format_exc()})
    else:
        logger.info('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = config['Settings']['PORT']

    model = joblib.load('finalized_model.sav')
    model_columns = joblib.load('model_columns.pkl')
    logger.info(str("API is running at " + "http://localhost:" + str(port) + "/predict"))
    app.run(port=port)
