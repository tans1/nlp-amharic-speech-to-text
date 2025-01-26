import os
import sys
from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './scripts')))
from test_model import Prediction

prediction = Prediction()

app = Flask(__name__)
CORS(app) 

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    return {
        "status": "sucess",
        "message": "Amharic Speech Recognition API",
    }

@app.route('/predict', methods=['GET', 'POST'])
def handle_upload():
    return prediction.handle_df_upload(request, secure_filename, app)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 33507))
    app.config['TEMPLATES_AUTO_RELOAD'] = False
    app.run(host='0.0.0.0', debug=False, port=port)