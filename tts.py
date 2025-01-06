from TTS.api import TTS
import json

# Load the JSON file
with open('speaker_list.json', 'r') as file:
    data = json.load(file)

def text_to_speech(text, path,speaker_id):
    tts = TTS(model_name="tts_models/en/vctk/vits")
    tts.tts_to_file(text=text, file_path=path, speaker=speaker_id)
    return path

input_text = "The happiest day of the year varies for everyone, but it is often a personal moment filled with joy, love, or achievement."
for speaker_id in data.keys():
    print(speaker_id)
    output_path = text_to_speech(input_text, path=f"Audio Results/{speaker_id}.mp3",speaker_id=speaker_id)
    print(speaker_id)