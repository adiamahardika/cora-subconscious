from flask import request, jsonify, Response
import os
from .. import ai_speech_bp
from dotenv import load_dotenv
from openai import OpenAI
import pyaudio
import datetime
import io

# Get the absolute path to the config file
# Get the directory of the current file

env_path = os.path.join(os.path.dirname(__file__), "../", ".env")

current_dir = os.path.dirname(__file__)

load_dotenv(dotenv_path=env_path)

openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_api_key)


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
        text_emotion = request.json.get('emotion')
        text_ai_mood = "Cheerful"
    else:
        return jsonify({'error': 'Invalid content type, expected application/json'}), 400

    if not text_gender:
        return jsonify({'error': 'Text is required'}), 400

    # Generate the greeting using OpenAI
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Robot assistant helping give guidance to your user"},
            {"role": "user", "content":   f"Generate a Greeting based on the person's gender which is {text_gender}"
             f"and time of day which is {
                 text_time}. Do not make it personal and make it "
             f"short and concise. Include a short prefix in the sentence based on the "
             f"person's gender. Don't use names or placeholders. For example: 'Good Morning Miss, "
             f"How can I help you?' or 'Good Morning Sir, How can I help you?'. "
             f"If the gender is They, assume that they are a part of a group and adjust the greeting based on this assumption, for example if a group appears, a greeting can be: Hello everyone, how can I help you?"
             f"If the gender of the user is They, do not use gender based acronym for groups of people such as gentlemen, sir, miss, or other similar words when the gender is they, also do not use words like team or other group adjectives and instead stick to more neutral words like everyone"
             f"The user's current emotion is {
                 text_emotion}. Adjust the message to match "
             f"the user's current mood. For example, if the user is feeling sad, cheer up "
             f"the user in the message. If the user is feeling neutral, keep the tone neutral."
             f"generate the response to the user with your current mood in mind, your current mood is {
                 text_ai_mood}"
             }
        ]
    )

    # Get the response text from LLM
    generated_text = completion.choices[0].message.content

  # Get the response text from LLM
    generated_text = completion.choices[0].message.content

    if not generated_text:
        return jsonify({'error': 'No response generated from LLM'}), 400

    # Ensure the logs directory exists
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Write the generated text to a log file
    log_file_path = os.path.join(logs_dir, "generated_greetings.log")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"[{timestamp}] {generated_text}\n")

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

    if not text:
        return jsonify({"error": "Text is required"}), 400

    # Choose voice based on gender
    voice = "alloy" if gender == "male" else "nova" if gender == "female" else "alloy"

    try:
        # Generate audio
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="wav"
        )
        buffer = io.BytesIO()

        # Accumulate the audio chunks into the buffer
        for chunk in response.iter_bytes(1024):
            buffer.write(chunk)

        # Return the audio as a single Blob
        buffer.seek(0)
        return Response(
            buffer,
            content_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=audio.wav"}
        )
    except Exception as e:
        print(f"Error generating audio: {e}")
        return jsonify({"error": str(e)}), 500
