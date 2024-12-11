from langchain_google_genai import ChatGoogleGenerativeAI
from flask import request, jsonify, Response
import os
import openai
import base64
from flask_socketio import SocketIO
from .. import ai_speech_bp
from dotenv import load_dotenv
from transformers import AutoProcessor, BarkModel
import scipy
from openai import OpenAI
import pyaudio

# Get the absolute path to the config file
# Get the directory of the current file

env_path = os.path.join(os.path.dirname(__file__), "../", ".env")

current_dir = os.path.dirname(__file__)

load_dotenv(dotenv_path=env_path)

config_dir_env = os.getenv('CONFIG_FILE_PATH')
openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_api_key)

config_dir = os.path.abspath(os.path.join(os.path.dirname(
    __file__), config_dir_env))



if not os.path.isfile(config_dir):
    raise Exception(f"Credentials file not found at {config_dir}")

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config_dir
# llm = ChatGoogleGenerativeAI(model="gemini-pro")


# # Load tokenizer and model
# model_name = "EleutherAI/gpt-neo-125M"  # Use other sizes if needed
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

@ai_speech_bp.route('/generate-greeting', methods=['POST'])
def synthesize_greeting():
    if request.is_json:
        text_gender = request.json.get('text')
        text_time = request.json.get('time')
    else:
        return jsonify({'error': 'Invalid content type, expected application/json'}), 400

    if not text_gender:
        return jsonify({'error': 'Text is required'}), 400
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Robot assistant helping give guidance to your user"},
            {"role": "user", "content": f"Generate a Greeting based on the person's gender which is {text_gender} and time of day which is {text_time}, do not make it personal and make it short and concise, include a short prefix in the sentence based on the person's gender, don't use names or placehodlers for example: Good Morning Miss, How can I help you? or Good Morning Sir, How can I help you?"}
        ]
    )

    # Get the response text from LLM
    generated_text = completion.choices[0].message.content
    
    print(generated_text)

    if not generated_text:
        return jsonify({'error': 'No response generated from LLM'}), 400
    
    # processor = AutoProcessor.from_pretrained("suno/bark")

    # model = BarkModel.from_pretrained("suno/bark")

    # voice_preset = "v2/en_speaker_6"

    # inputs = processor(generated_text, voice_preset=voice_preset)

    # audio_array = model.generate(**inputs)
    # audio_array = audio_array.cpu().numpy().squeeze()
    
    # sample_rate = model.generation_config.sample_rate   

    # scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)


    # Return the generated text to the frontend
    return jsonify({"text": generated_text})


# Initialize PyAudio for real-time playback (local playback)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,  # PCM format
                channels=1,
                rate=24000,  # Same as OpenAI TTS default
                output=True)


@ai_speech_bp.route('/generate-audio', methods=['POST'])
def generate_audio():
    data = request.json
    text = data.get("text")
    gender = data.get("gender")
    
    print (gender)

    if not text:
        return jsonify({"error": "Text is required"}), 400

    voice = "alloy" if gender == "male" else "nova" if gender == "female" else "alloy"
    
    print (voice)

    if not voice:
        return jsonify({"error": "Invalid gender"}), 400

    def audio_stream():
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                response_format="wav",  # Raw audio format
            )
            for chunk in response.iter_bytes(1024):
                yield chunk
        except Exception as e:
            print(f"Error generating audio: {e}")
            return  # Stop the generator on error
        
        
    return Response(audio_stream(), content_type="audio/wav")
