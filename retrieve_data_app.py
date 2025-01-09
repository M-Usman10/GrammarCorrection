from flask import Flask, render_template, jsonify, send_file, request
from utils.database_utils import MongoDBClient
import gridfs
import io

app = Flask(__name__)

# MongoDB Client
mongo_client = MongoDBClient()


@app.route("/")
def index():
    """
    Render the main UI to display records and audio files.
    """
    try:
        records = list(mongo_client.db["records"].find())
        return render_template("db_index.html", records=records)
    except Exception as e:
        return f"Error retrieving data: {str(e)}"


@app.route("/get_audio/<file_id>", methods=["GET"])
def get_audio(file_id):
    """
    Retrieve and return audio file for playback.
    """
    try:
        file_bytes = mongo_client.get_file(file_id)
        return send_file(
            io.BytesIO(file_bytes),
            mimetype="audio/wav",
            as_attachment=False,
            download_name=f"{file_id}.wav",
        )
    except gridfs.errors.NoFile:
        return jsonify({"error": "Audio file not found in database."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)