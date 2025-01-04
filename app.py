# app.py
import os
from flask import Flask, render_template, request, jsonify
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput

from utils import (
    list_models,
    create_model,
    delete_model,
    start_triton_server,
    stop_triton_server,
    get_triton_logs,
    sanitize_model_name
)

app = Flask(__name__)

TRITON_SERVER_URL = "localhost:8000"

def infer_model(model_name, input_text):
    """
    Inference logic with special shape for T5, else default.
    """
    client = InferenceServerClient(url=TRITON_SERVER_URL, network_timeout=1000)

    # Example T5 grammar corrector
    if model_name == "deep-learning-analytics-Grammar-Correction-T5":
        inputs = [InferInput("DUMMY_INPUT", [1, 1, 1], "BYTES")]
        inputs[0].set_data_from_numpy(np.array([[[input_text]]], dtype=object))
    else:
        # Default shape [1,1]
        inputs = [InferInput("INPUT", [1, 1], "BYTES")]
        inputs[0].set_data_from_numpy(np.array([[input_text]], dtype=object))

    response = client.infer(model_name, inputs)
    return response.as_numpy("OUTPUT")

@app.route('/')
def index():
    # Single page for everything
    return render_template('index.html')

# ------------------ AJAX ENDPOINTS ------------------

@app.route('/api/list_models', methods=['GET'])
def api_list_models():
    """
    Return the list of model folders in ./models
    """
    models = list_models("./models")
    return jsonify({'models': models})

@app.route('/api/query', methods=['POST'])
def query_model():
    """
    Inference endpoint
    """
    model_name = request.form.get('model', '').strip()
    input_text = request.form.get('query', '')
    try:
        output = infer_model(model_name, input_text)
        if output.size > 0:
            decoded = output[0].decode('utf-8')
        else:
            decoded = ""
        return jsonify({'response': decoded})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/add_model', methods=['POST'])
def api_add_model():
    """
    Create a new model (folder) with sanitized name.
    """
    new_model_name = request.form.get('new_model_name', '').strip()
    if not new_model_name:
        return jsonify({'error': "Model name cannot be empty."}), 400
    try:
        create_model(new_model_name, "./models")
        safe_name = sanitize_model_name(new_model_name)
        usage_code = f"""import numpy as np
from tritonclient.http import InferenceServerClient, InferInput

TRITON_SERVER_URL = "localhost:8000"
client = InferenceServerClient(url=TRITON_SERVER_URL, network_timeout=1000)

# Prepare input for model
user_command = "Hi, my name is Samuel"
inputs_model = [InferInput("INPUT", [1, 1], "BYTES")]
inputs_model[0].set_data_from_numpy(np.array([[user_command]], dtype=object))

response_model = client.infer("{safe_name}", inputs_model)
model_result = response_model.as_numpy("OUTPUT")

print("Model Response:", model_result)
"""
        return jsonify({
            'status': 'success',
            'message': f"Model '{new_model_name}' created successfully (folder: {safe_name}).",
            'usage_code': usage_code
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_model', methods=['POST'])
def api_delete_model():
    """
    Delete a model directory from ./models
    """
    model_folder_name = request.form.get('delete_model_name', '').strip()
    if not model_folder_name:
        return jsonify({'error': "Model name cannot be empty."}), 400
    try:
        delete_model(model_folder_name, "./models")
        return jsonify({'status': 'success', 'message': f"Model '{model_folder_name}' deleted successfully."})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_client_code', methods=['POST'])
def api_get_client_code():
    """
    Return the usage code snippet for an existing sanitized model folder.
    We don't know the original name, so we just assume user wants to
    call Triton with the sanitized name.
    """
    folder_name = request.form.get('folder_name', '').strip()
    if not folder_name:
        return jsonify({'error': "Folder name cannot be empty."}), 400
    # Generate code snippet for this folder
    usage_code = f"""import numpy as np
from tritonclient.http import InferenceServerClient, InferInput

TRITON_SERVER_URL = "localhost:8000"
client = InferenceServerClient(url=TRITON_SERVER_URL, network_timeout=1000)

# Prepare input for model
user_command = "Hi, my name is Samuel"
inputs_model = [InferInput("INPUT", [1, 1], "BYTES")]
inputs_model[0].set_data_from_numpy(np.array([[user_command]], dtype=object))

response_model = client.infer("{folder_name}", inputs_model)
model_result = response_model.as_numpy("OUTPUT")

print("Model Response:", model_result)
"""
    return jsonify({
        'status': 'success',
        'usage_code': usage_code
    })

@app.route('/api/start_server', methods=['POST'])
def api_start_server():
    success, msg = start_triton_server()
    if success:
        return jsonify({'status': 'success', 'message': msg})
    return jsonify({'error': msg}), 500

@app.route('/api/stop_server', methods=['POST'])
def api_stop_server():
    success, msg = stop_triton_server()
    if success:
        return jsonify({'status': 'success', 'message': msg})
    return jsonify({'error': msg}), 500

@app.route('/api/get_server_logs', methods=['GET'])
def api_get_server_logs():
    logs = get_triton_logs()
    return jsonify({'logs': logs})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000,debug=True)