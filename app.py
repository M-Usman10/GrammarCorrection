import os
import io
import tempfile
import subprocess
import re
import datetime
import base64
from flask import Flask, render_template, request, jsonify
import numpy as np
from tritonclient.http import InferenceServerClient, InferInput
from TTS.api import TTS
from bson import ObjectId

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

# Start Ollama server if not running
if not subprocess.run(["pgrep", "-f", "ollama serve"], stdout=subprocess.DEVNULL).returncode == 0:
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

mongo_client = MongoDBClient()

# Hardcode for now
USERNAME = "samuel"

# -----------------------------------------------------
# ID Generators
# -----------------------------------------------------
def generate_record_id():
    """
    Format: yyddmmhHH
    E.g. 251205h09
    """
    now_utc = datetime.datetime.utcnow()
    return now_utc.strftime("%y%m%d") + "h" + now_utc.strftime("%H")

def generate_transaction_id():
    """
    Format: yymmdd_hhmmss + 'usrc'
    E.g. 251205_093012usrc
    """
    now_utc = datetime.datetime.utcnow()
    base_str = now_utc.strftime("%y%m%d_%H%M%S")
    return base_str + "usrc"

# -----------------------------------------------------
# TTS text sanitizer (remove triple backticks)
# -----------------------------------------------------
def sanitize_tts_text(text: str) -> str:
    """
    Remove triple backticks plus some other symbols,
    but keep single backticks if needed.
    """
    text = re.sub(r"```+", "", text)
    pattern = r"[#\*\_>~\-\+\=\|\{\}\[\]\(\)]"
    return re.sub(pattern, "", text)

# -----------------------------------------------------
# TRITON inference
# -----------------------------------------------------
def infer_model(model_name, chat_history_or_text):
    grammar_models = [
        "happytransformer",
        "grammarly-coedit-large",
        "deep-learning-analytics-Grammar-Correction-T5"
    ]
    client = InferenceServerClient(url=TRITON_SERVER_URL, network_timeout=1000)
    if model_name in grammar_models:
        inputs = [InferInput("INPUT", [1, 1], "BYTES")]
        inputs[0].set_data_from_numpy(np.array([[chat_history_or_text]], dtype=object))
    else:
        import json
        as_json = json.dumps(chat_history_or_text)
        inputs = [InferInput("INPUT", [1, 1], "BYTES")]
        inputs[0].set_data_from_numpy(np.array([[as_json]], dtype=object))

    response = client.infer(model_name, inputs)
    return response.as_numpy("OUTPUT")

# -----------------------------------------------------
# Audio conversion
# -----------------------------------------------------
def convert_audio_to_wav(input_bytes: bytes, filename: str) -> bytes:
    extension = os.path.splitext(filename)[1].lower()
    if not extension:
        extension = ".webm"
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_input:
        temp_input.write(input_bytes)
        temp_input.flush()
        temp_input_path = temp_input.name

    temp_wav_path = temp_input_path + ".wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_input_path,
        "-ar", "16000",
        "-ac", "1",
        temp_wav_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e}")
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

    with open(temp_wav_path, "rb") as f:
        wav_bytes = f.read()
    os.remove(temp_wav_path)
    return wav_bytes

def transcribe_whisper(wav_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        temp_path = tmp.name

    import librosa
    audio_data, _ = librosa.load(temp_path, sr=16000, mono=True)
    audio_data = audio_data.astype(np.float32)
    os.remove(temp_path)

    client = InferenceServerClient(url=TRITON_SERVER_URL, network_timeout=500)
    inputs = [InferInput("AUDIO_INPUT", [len(audio_data)], "FP32")]
    inputs[0].set_data_from_numpy(audio_data)

    response = client.infer("whisper", inputs)
    result = response.as_numpy("OUTPUT")
    if result is not None and result.size > 0:
        return result[0].decode("utf-8")
    return ""

# -----------------------------------------------------
# ROUTES
# -----------------------------------------------------

@app.route('/')
def index():
    return render_template('umisource.html')


@app.route('/api/new_transaction', methods=['POST'])
def new_transaction():
    """
    New structure:
    {
      "umisource": {
        "<username>": {
          "<yy>": {
            "<yyddmmhHH>": {
              "<unique_id_for_transaction>": {
                "transaction id": "<yymmdd_hhmmssusrc>",
                "chat_history": [],
                "status": "open",
                "created_at": ...
              }
            }
          }
        }
      }
    }
    """
    now_utc = datetime.datetime.utcnow()
    yy = now_utc.strftime("%y")
    rec_id = generate_record_id()
    trans_id = generate_transaction_id()

    inserted_id = mongo_client.create_new_transaction(USERNAME, yy, rec_id, trans_id)
    return jsonify({
        "year": yy,
        "record_id": rec_id,
        "transaction_id": trans_id,
        "_id": str(inserted_id)
    })

@app.route('/api/cancel_transaction', methods=['POST'])
def cancel_transaction():
    doc_id = request.form.get('_id', '').strip()
    if not doc_id:
        return jsonify({"error": "No _id provided"}), 400
    doc = mongo_client.get_transaction_by_oid(doc_id)
    if not doc:
        return jsonify({"error": "Doc not found"}),404

    # find the single unique transaction
    # doc["umisource"][USERNAME][yy][rec_id] => dict of { uniqueKey: { } }
    # We'll find the 1 uniqueKey
    # If no chat_history => delete doc; else set status=canceled
    try:
        um_data = doc["umisource"][USERNAME]
        # might be multiple years, but assume only 1
        for year_key in um_data.keys():
            rec_dict = um_data[year_key]
            for rec_key in rec_dict.keys():
                unique_map = rec_dict[rec_key]
                for unique_key in unique_map:
                    tdict = unique_map[unique_key]
                    # check chat_history
                    ch = tdict.get("chat_history", [])
                    if len(ch)==0:
                        mongo_client.delete_transaction_by_oid(doc_id)
                        return jsonify({"status":"deleted"})
                    else:
                        # set status => canceled
                        mongo_client.update_transaction_status(doc_id, USERNAME, year_key, rec_key, unique_key, "canceled")
                        return jsonify({"status":"canceled"})
    except:
        # if structure is not as expected => delete
        mongo_client.delete_transaction_by_oid(doc_id)
        return jsonify({"status":"deleted"})

    return jsonify({"status":"error","msg":"No transaction found in doc"})

@app.route('/api/find_transactions_by_tid', methods=['GET'])
def find_transactions_by_tid():
    t_id = request.args.get('transaction_id','').strip()
    if not t_id:
        return jsonify({"error":"No transaction_id provided"}),400

    docs = mongo_client.find_transactions_by_tid(t_id)
    out=[]
    for d in docs:
        # We'll parse doc => find the exact transaction
        try:
            um_data = d["umisource"][USERNAME]
            for year_key in um_data:
                rec_dict = um_data[year_key]
                for rec_key in rec_dict:
                    unique_map = rec_dict[rec_key]
                    for unique_id, tdict in unique_map.items():
                        if tdict.get("transaction id")== t_id:
                            created_at = tdict.get("created_at")
                            if hasattr(created_at, "isoformat"):
                                created_at = created_at.isoformat()
                            out.append({
                                "_id": str(d["_id"]),
                                "transaction_id": t_id,
                                "record_id": rec_key,
                                "year": year_key,
                                "created_at": created_at
                            })
        except:
            pass
    return jsonify({"results": out})

@app.route('/api/load_transaction', methods=['GET'])
def load_transaction():
    doc_id = request.args.get('_id','').strip()
    if not doc_id:
        return jsonify({"error":"No _id provided"}),400
    doc = mongo_client.get_transaction_by_oid(doc_id)
    if not doc:
        return jsonify({"error":"Transaction not found"}),404

    # find the 1 transaction
    # We flatten => doc["transaction_id"], doc["chat_history"]
    # if multiple, just pick first
    try:
        um_data = doc["umisource"][USERNAME]
        for year_key in um_data:
            rec_dict = um_data[year_key]
            for rec_key in rec_dict:
                unique_map = rec_dict[rec_key]
                for unique_id, tdict in unique_map.items():
                    chat_hist = tdict.get("chat_history", [])
                    doc["transaction_id"] = tdict["transaction id"]
                    doc["chat_history"] = chat_hist
                    doc["_id"] = str(doc["_id"])
                    return jsonify(doc)
    except:
        pass

    doc["_id"] = str(doc["_id"])
    doc["transaction_id"]=""
    doc["chat_history"]=[]
    return jsonify(doc)

@app.route('/api/get_chat_history', methods=['GET'])
def get_chat_history():
    doc_id = request.args.get('_id','').strip()
    if not doc_id:
        return jsonify({"error":"No _id provided"}),400
    doc = mongo_client.get_transaction_by_oid(doc_id)
    if not doc:
        return jsonify({"error":"Transaction not found"}),404

    try:
        um_data = doc["umisource"][USERNAME]
        for yk in um_data:
            rec_dict = um_data[yk]
            for rk in rec_dict:
                unique_map = rec_dict[rk]
                for uk, tdict in unique_map.items():
                    t_id = tdict.get("transaction id","")
                    ch = tdict.get("chat_history",[])
                    return jsonify({"chat_history":ch,"transaction_id": t_id})
    except:
        pass

    return jsonify({"chat_history":[],"transaction_id":""})

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    file = request.files.get('audio_data')
    doc_id = request.form.get('_id','').strip()
    if not file or not doc_id:
        return jsonify({"error":"Missing file or _id"}),400
    try:
        input_bytes = file.read()
        filename = file.filename or "recording.webm"
        wav_bytes = convert_audio_to_wav(input_bytes, filename)
        file_id = mongo_client.store_file(
            file_bytes=wav_bytes,
            filename="recording.wav",
            content_type="audio/wav",
            metadata={"purpose":"speech-to-text"}
        )
        text_out = transcribe_whisper(wav_bytes)
        # append user stt
        mongo_client.append_chat_message_stt(doc_id, text_out, file_id, USERNAME)
        return jsonify({'transcription': text_out})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route('/api/query', methods=['POST'])
def query_model():
    model_name = request.form.get('model','').strip()
    user_prompt = request.form.get('query','')
    doc_id = request.form.get('_id','').strip()
    system_prompt = request.form.get('system_prompt','').strip()

    if not doc_id:
        return jsonify({"error":"No _id provided."}),400
    if not model_name:
        return jsonify({"error":"No model name provided."}),400

    doc = mongo_client.get_transaction_by_oid(doc_id)
    if not doc:
        return jsonify({"error":f"No doc found by _id={doc_id}"}),404

    # extract the single transaction
    # append system if needed
    # append user
    # build conversation => skip TTS or STT
    try:
        um_data = doc["umisource"][USERNAME]
        for yk in um_data:
            rec_dict = um_data[yk]
            for rk in rec_dict:
                unique_map = rec_dict[rk]
                for uk, tdict in unique_map.items():
                    ch = tdict.get("chat_history",[])
                    # system
                    last_sys = None
                    for msg in reversed(ch):
                        if msg.get("role")=="system":
                            last_sys = msg["content"]
                            break
                    if system_prompt and system_prompt!=last_sys:
                        ch.append({
                            "role":"system",
                            "content":system_prompt,
                            "interaction_type":"system",
                            "timestamp": datetime.datetime.utcnow()
                        })
                    if user_prompt.strip():
                        ch.append({
                            "role":"user",
                            "content": user_prompt,
                            "interaction_type":"text-to-text",
                            "timestamp": datetime.datetime.utcnow()
                        })
                    else:
                        return jsonify({"error":"No user prompt provided."}),400

                    # build final conversation
                    grammar = ["happytransformer","grammarly-coedit-large","deep-learning-analytics-Grammar-Correction-T5"]
                    if model_name in grammar:
                        to_infer = user_prompt
                    else:
                        conv = []
                        for msg in ch:
                            itype = msg.get("interaction_type","")
                            if itype in ["speech-to-text","text-to-speech"]:
                                continue
                            r = msg["role"]
                            if r.startswith("openai") or r.startswith("ollama") or r.startswith("gpt") or r in grammar:
                                conv.append({"role":"assistant","content":msg["content"]})
                            elif r=="system":
                                conv.append({"role":"system","content":msg["content"]})
                            elif r=="user":
                                conv.append({"role":"user","content":msg["content"]})
                            else:
                                conv.append({"role":r,"content":msg["content"]})
                        to_infer = conv

                    raw_out = infer_model(model_name, to_infer)
                    decoded = raw_out[0].decode('utf-8') if raw_out.size>0 else ""
                    ch.append({
                        "role": model_name,
                        "content": decoded,
                        "interaction_type":"text-to-text",
                        "timestamp": datetime.datetime.utcnow()
                    })
                    tdict["chat_history"] = ch
                    # if there's at least 1 => closed
                    tdict["status"]="closed"

                    # update
                    mongo_client.update_transaction_subdict(doc_id, USERNAME, yk, rk, uk, tdict)
                    return jsonify({"response": decoded})
    except Exception as e:
        return jsonify({"error":str(e)}),400

    return jsonify({"error":"No transaction found in doc structure"}),404

@app.route('/api/tts', methods=['POST'])
def tts_endpoint():
    text = request.form.get('text','')
    speaker = request.form.get('speaker','p229')
    speed_str = request.form.get('speed','1.0')
    doc_id = request.form.get('_id','').strip()
    if not text:
        return jsonify({'error':'No text provided.'}),400

    clean_text = sanitize_tts_text(text)
    try:
        speed = float(speed_str)
        speed = max(0.5, min(speed,2.0))

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name

        local_tts = TTS(model_name="tts_models/en/vctk/vits")
        local_tts.tts_to_file(text=clean_text, file_path=tmp_path, speaker=speaker)

        if abs(speed-1.0)>1e-5:
            tmp2 = tmp_path+".speed.wav"
            cmd=["ffmpeg","-y","-i",tmp_path,"-filter:a",f"atempo={speed}", tmp2]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            with open(tmp2,'rb') as f:
                raw_audio = f.read()
            os.remove(tmp2)
            os.remove(tmp_path)
        else:
            with open(tmp_path,'rb') as f:
                raw_audio = f.read()
            os.remove(tmp_path)

        file_id = mongo_client.store_file(
            file_bytes=raw_audio,
            filename="tts_output.wav",
            content_type="audio/wav",
            metadata={"purpose":"text-to-speech","speaker":speaker,"speed":speed}
        )

        if doc_id:
            doc = mongo_client.get_transaction_by_oid(doc_id)
            if doc:
                # find the single transaction
                try:
                    um_data = doc["umisource"][USERNAME]
                    for yk in um_data:
                        rec_dict = um_data[yk]
                        for rk in rec_dict:
                            unique_map = rec_dict[rk]
                            for uk, tdict in unique_map.items():
                                ch = tdict.get("chat_history",[])
                                ch.append({
                                    "role":"tts_engine",
                                    "content":text,
                                    "interaction_type":"text-to-speech",
                                    "audio_file_id":file_id,
                                    "timestamp":datetime.datetime.utcnow()
                                })
                                tdict["chat_history"]=ch
                                mongo_client.update_transaction_subdict(doc_id, USERNAME, yk, rk, uk, tdict)
                                break
                except:
                    pass

        return jsonify({'audio': base64.b64encode(raw_audio).decode('utf-8')})
    except Exception as e:
        return jsonify({"error":str(e)}),500

# Model management
@app.route('/api/list_models', methods=['GET'])
def api_list_models():
    mods = list_models("./models")
    return jsonify({'models':mods})

@app.route('/api/add_model', methods=['POST'])
def api_add_model():
    nm= request.form.get('new_model_name','').strip()
    if not nm: return jsonify({'error':"Model name cannot be empty."}),400
    try:
        create_model(nm, "./models")
        sn = sanitize_model_name(nm)
        usage_code=f"""import numpy as np
from tritonclient.http import InferenceServerClient, InferInput

TRITON_SERVER_URL="localhost:8000"
client=InferenceServerClient(url=TRITON_SERVER_URL, network_timeout=1000)
user_command="Hi, my name is Samuel"
inp=[InferInput("INPUT",[1,1],"BYTES")]
inp[0].set_data_from_numpy(np.array([[user_command]],dtype=object))
rsp=client.infer("{sn}", inp)
model_result=rsp.as_numpy("OUTPUT")
print("Model Response:", model_result)
"""
        return jsonify({
            'status':"success",
            'message':f"Model '{nm}' created successfully (folder:{sn}).",
            'usage_code': usage_code
        })
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/delete_model', methods=['POST'])
def api_delete_model():
    mfn= request.form.get('delete_model_name','').strip()
    if not mfn: return jsonify({'error':"Model name cannot be empty."}),400
    try:
        delete_model(mfn,"./models")
        return jsonify({'status':"success",'message':f"Model '{mfn}' deleted successfully."})
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/get_client_code', methods=['POST'])
def api_get_client_code():
    fn= request.form.get('folder_name','').strip()
    if not fn: return jsonify({'error':"Folder name cannot be empty."}),400
    usage_code=f"""import numpy as np
from tritonclient.http import InferenceServerClient, InferInput

TRITON_SERVER_URL="localhost:8000"
client=InferenceServerClient(url=TRITON_SERVER_URL,network_timeout=1000)
user_command="Hi, my name is Samuel"
inp=[InferInput("INPUT",[1,1],"BYTES")]
inp[0].set_data_from_numpy(np.array([[user_command]],dtype=object))
rsp=client.infer("{fn}", inp)
model_result=rsp.as_numpy("OUTPUT")
print("Model Response:", model_result)
"""
    return jsonify({'status':"success",'usage_code':usage_code})

# server logs
@app.route('/api/get_server_logs', methods=['GET'])
def api_get_server_logs():
    logs=get_triton_logs()
    return jsonify({'logs':logs})

@app.route('/api/start_server', methods=['POST'])
def api_start_server():
    suc,msg,lg = start_triton_server()
    if suc:
        return jsonify({'status':"success",'message':msg})
    return jsonify({'error':msg,'logs':lg}),500

@app.route('/api/stop_server', methods=['POST'])
def api_stop_server():
    s,m= stop_triton_server()
    if s: return jsonify({'status':"success",'message':m})
    return jsonify({'error':m}),500

@app.route('/api/get_file/<file_id>',methods=['GET'])
def api_get_file(file_id):
    try:
        fb= mongo_client.get_file(file_id)
        b64= base64.b64encode(fb).decode('utf-8')
        return jsonify({"base64_data":b64})
    except Exception as e:
        return jsonify({"error":str(e)}),400

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
