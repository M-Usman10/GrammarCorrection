# app.py
from flask import Flask, render_template, request, jsonify
from tritonclient.http import InferenceServerClient, InferInput
import numpy as np

app = Flask(__name__)

TRITON_SERVER_URL = "localhost:8000"

def infer_model(model_name, input_text):
    # Create a new client instance for each request to avoid thread issues
    client = InferenceServerClient(url=TRITON_SERVER_URL, network_timeout=1000)

    # Special case for T5 Grammar Correction model (uses "DUMMY_INPUT")
    if model_name == "deep-learning-analytics-Grammar-Correction-T5":
        input_tensor = [InferInput("DUMMY_INPUT", [1, 1, 1], "BYTES")]
        input_tensor[0].set_data_from_numpy(np.array([[[input_text]]], dtype=object))
    else:
        # Other models use "INPUT"
        input_tensor = [InferInput("INPUT", [1, 1], "BYTES")]
        input_tensor[0].set_data_from_numpy(np.array([[input_text]], dtype=object))

    response = client.infer(model_name, input_tensor)
    return response.as_numpy("OUTPUT")

@app.route('/')
def index():
    # By default, let's load variant #1.
    # You can switch to variant2.html or variant3.html in the next lines.
    return render_template('Google-Like.html')

@app.route('/query', methods=['POST'])
def query_model():
    model_name = request.form['model']
    input_text = request.form['query']
    try:
        response = infer_model(model_name, input_text)
        # decode the response if it's bytes
        decoded_response = response[0].decode('utf-8') if response.size > 0 else ""
        return jsonify({'response': decoded_response})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)