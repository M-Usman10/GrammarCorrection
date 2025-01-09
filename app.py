import os
import io
import tempfile
import subprocess
import base64
import numpy as np
import librosa
import random
import string
from flask import Flask, render_template, request, jsonify
from tritonclient.http import InferenceServerClient, InferInput
from TTS.api import TTS

# Import our new utils modules
from utils.triton_server_utils import (
    start_triton_server, stop_triton_server, get_triton_logs
)
from utils.model_utils import (
    list_models, create_model, delete_model,
    sanitize_model_name
)
from utils.database_utils import MongoDBClient

app = Flask(__name__)

TRITON_SERVER_URL = "localhost:8000"

# Start Ollama server if not running (same logic as before)
if not subprocess.run(["pgrep", "-f", "ollama serve"], stdout=subprocess.DEVNULL).returncode == 0:
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Create a global MongoDB client
mongo_client = MongoDBClient()


def generate_activity_id(length=8):
    """Generate a random alpha-numeric activity ID (max 8 characters)."""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def get_next_step_index(activity_id):
    """
    Find the highest step_index in the DB for this activity_id
    and return step_index + 1. If none found, return 0.
    """
    records = mongo_client.get_records_by_activity_id(activity_id)
    if not records:
        return 0
    max_step = max(r.get("step_index", 0) for r in records)
    return max_step + 1


def infer_model(model_name, input_text):
    client = InferenceServerClient(url=TRITON_SERVER_URL, network_timeout=1000)
    # Example special-casing for T5
    if model_name == "deep-learning-analytics-Grammar-Correction-T5":
        inputs = [InferInput("DUMMY_INPUT", [1, 1, 1], "BYTES")]
        inputs[0].set_data_from_numpy(np.array([[[input_text]]], dtype=object))
    else:
        inputs = [InferInput("INPUT", [1, 1], "BYTES")]
        inputs[0].set_data_from_numpy(np.array([[input_text]], dtype=object))

    response = client.infer(model_name, inputs)
    return response.as_numpy("OUTPUT")


def convert_webm_to_wav(webm_bytes) -> bytes:
    """
    Convert in-memory WebM bytes to WAV bytes, returning the WAV content.
    """
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_input:
        temp_input.write(webm_bytes)
        temp_input.flush()
        temp_webm_path = temp_input.name

    temp_wav_path = temp_webm_path.replace(".webm", ".wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_webm_path,
        "-ar", "16000",
        "-ac", "1",
        temp_wav_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e}")
    finally:
        if os.path.exists(temp_webm_path):
            os.remove(temp_webm_path)

    with open(temp_wav_path, "rb") as f:
        wav_bytes = f.read()
    os.remove(temp_wav_path)
    return wav_bytes


def transcribe_whisper(wav_bytes: bytes):
    """
    Send WAV bytes to the 'whisper' model on Triton to get transcription text.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        temp_path = tmp.name

    audio_data, _ = librosa.load(temp_path, sr=16000, mono=True)
    audio_data = audio_data.astype(np.float32)
    os.remove(temp_path)

    client = InferenceServerClient(url=TRITON_SERVER_URL, network_timeout=500)
    inputs = [InferInput("AUDIO_INPUT", [len(audio_data)], "FP32")]
    inputs[0].set_data_from_numpy(audio_data)

    response = client.infer("whisper", inputs)
    transcription = response.as_numpy("OUTPUT")
    if transcription is not None and transcription.size > 0:
        return transcription[0].decode("utf-8")
    return ""


@app.route('/')
def index():
    return render_template('Google-Like.html')


@app.route('/api/new_activity', methods=['GET'])
def new_activity():
    """Generate and return a new activity ID."""
    activity_id = generate_activity_id()
    return jsonify({"activity_id": activity_id})


@app.route('/api/list_models', methods=['GET'])
def api_list_models():
    models = list_models("./models")
    return jsonify({'models': models})


@app.route('/api/query', methods=['POST'])
def query_model():
    """
    Model inference for user-provided text.
    We also store the successful interaction in DB if no error.
    Expects 'activity_id' in the form.
    """
    model_name = request.form.get('model', '').strip()
    input_text = request.form.get('query', '')
    activity_id = request.form.get('activity_id', '').strip()
    try:
        output = infer_model(model_name, input_text)
        decoded = output[0].decode('utf-8') if output.size > 0 else ""

        # Determine step_index for this new record
        step_index = get_next_step_index(activity_id)

        mongo_client.store_interaction(
            interaction_type="text-to-text",
            input_data=input_text,
            output_data=decoded,
            model_name=model_name,
            metadata={"info": "query endpoint"},
            activity_id=activity_id,
            step_index=step_index
        )

        return jsonify({'response': decoded})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Accepts 'audio_data' in WebM format, converts to WAV,
    transcribes via Whisper, stores record in DB.
    Expects 'activity_id' in the form data.
    """
    file = request.files.get('audio_data')
    activity_id = request.form.get('activity_id', '').strip()
    if not file:
        return jsonify({'error': 'No audio file received.'}), 400

    try:
        webm_bytes = file.read()
        wav_bytes = convert_webm_to_wav(webm_bytes)

        # Store WAV in DB (GridFS)
        file_id = mongo_client.store_file(
            file_bytes=wav_bytes,
            filename="recording.wav",
            content_type="audio/wav",
            metadata={"purpose": "speech-to-text"}
        )

        # Transcribe
        transcription_text = transcribe_whisper(wav_bytes)

        # Determine step_index for this new record
        step_index = get_next_step_index(activity_id)

        mongo_client.store_interaction(
            interaction_type="speech-to-text",
            input_data=f"(WAV file in DB) file_id={file_id}",
            output_data=transcription_text,
            model_name="whisper",
            activity_id=activity_id,
            step_index=step_index
        )

        return jsonify({'transcription': transcription_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tts', methods=['POST'])
def tts_endpoint():
    """
    Convert text to speech using TTS, store the result in DB if desired,
    now also using activity_id if provided.
    """
    text = request.form.get('text', '')
    speaker = request.form.get('speaker', 'p229')
    speed_str = request.form.get('speed', '1.0')
    activity_id = request.form.get('activity_id', '').strip()  # Updated
    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    try:
        speed = float(speed_str)
        if speed < 0.5:
            speed = 0.5
        if speed > 2.0:
            speed = 2.0

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        local_tts = TTS(model_name="tts_models/en/vctk/vits")
        local_tts.tts_to_file(text=text, file_path=tmp_path, speaker=speaker)

        # Adjust speed if needed
        if abs(speed - 1.0) > 1e-5:
            tmp_output_path = tmp_path + '.speed.wav'
            cmd = [
                "ffmpeg", "-y",
                "-i", tmp_path,
                "-filter:a", f"atempo={speed}",
                tmp_output_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(tmp_output_path, 'rb') as f:
                raw_audio = f.read()
            os.remove(tmp_output_path)
            os.remove(tmp_path)
        else:
            with open(tmp_path, 'rb') as f:
                raw_audio = f.read()
            os.remove(tmp_path)

        # Optionally store the TTS result in DB
        file_id = mongo_client.store_file(
            file_bytes=raw_audio,
            filename="tts_output.wav",
            content_type="audio/wav",
            metadata={"purpose": "text-to-speech", "speaker": speaker, "speed": speed}
        )
        # Also store text record, attaching the same activity_id
        step_index = get_next_step_index(activity_id)
        mongo_client.store_interaction(
            interaction_type="text-to-speech",
            input_data=text,
            output_data=f"(TTS audio file in DB) file_id={file_id}",
            model_name="vits_tts",
            activity_id=activity_id,
            step_index=step_index
        )

        base64_audio = base64.b64encode(raw_audio).decode('utf-8')
        return jsonify({'audio': base64_audio})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Add & Delete model
@app.route('/api/add_model', methods=['POST'])
def api_add_model():
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
    folder_name = request.form.get('folder_name', '').strip()
    if not folder_name:
        return jsonify({'error': "Folder name cannot be empty."}), 400
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


# Get server logs
@app.route('/api/get_server_logs', methods=['GET'])
def api_get_server_logs():
    logs = get_triton_logs()
    return jsonify({'logs': logs})


# Start/Stop Triton server
@app.route('/api/start_server', methods=['POST'])
def api_start_server():
    success, msg, logs = start_triton_server()
    if success:
        return jsonify({'status': 'success', 'message': msg})
    return jsonify({'error': msg, 'logs': logs}), 500


@app.route('/api/stop_server', methods=['POST'])
def api_stop_server():
    success, msg = stop_triton_server()
    if success:
        return jsonify({'status': 'success', 'message': msg})
    return jsonify({'error': msg}), 500


# Example to get a file from DB if needed
@app.route('/api/get_file/<file_id>', methods=['GET'])
def api_get_file(file_id):
    """
    Retrieve a file from GridFS by file_id and return as raw bytes or base64.
    """
    try:
        file_bytes = mongo_client.get_file(file_id)
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        return jsonify({"base64_data": b64})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/get_activity', methods=['GET'])
def api_get_activity():
    """
    Return all interactions for a given activity_id, sorted by step_index ascending.
    """
    activity_id = request.args.get('activity_id', '').strip()
    if not activity_id:
        return jsonify({'error': 'No activity_id provided.'}), 400

    records = mongo_client.get_records_by_activity_id(activity_id)
    # Sort by step_index ascending
    sorted_records = sorted(records, key=lambda r: r.get("step_index", 0))
    out = []
    for r in sorted_records:
        r['_id'] = str(r['_id'])
        out.append(r)
    return jsonify({'records': out})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)