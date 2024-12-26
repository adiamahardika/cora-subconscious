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
        text_ai_mood = request.json.get('tone')
    else:
        return jsonify({'error': 'Invalid content type, expected application/json'}), 400

    if not text_gender:
        return jsonify({'error': 'Text is required'}), 400

    print(text_ai_mood)
    
    # Generate the greeting using OpenAI
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    # "You are an AI designed to generate unique greetings based on specific parameters. "
                    # "Your task is to create a personalized greeting that incorporates the following elements:\n\n"
                    # "1. **Time of Day**: Determine whether it is morning, afternoon, evening, or night.\n"
                    # "2. **Gender of the User**: Identify if the user is male, female, or a group (if multiple users are indicated).\n"
                    # "3. **User Mood**: Assess the mood of the user, which can be happy, sad, excited, indifferent, or any other relevant emotional state.\n"
                    # "4. **Preset Mood of the AI**: Consider the preset mood of the AI, which could be cheerful, neutral, empathetic, or formal.\n\n"
                    # "Use the information above to craft a greeting that is warm, engaging, and appropriate for the context. "
                    # "Ensure that the greeting reflects the time of day and aligns with the user's mood while also incorporating the AI's mood. "
                    # "The greeting should feel natural and personalized, avoiding clichés and generic phrases.\n\n"
                    # "**Example Structure for the Greeting**:\n"
                    # "- Greeting (e.g., 'Good morning!')\n"
                    # "- Acknowledgment of the user's mood (e.g., 'I hope you're feeling great today!')\n"
                    # "- Personal touch based on the user's gender or group (e.g., 'It's wonderful to see you all!')\n"
                    # "- Closing (e.g. 'How can I help you today?')\n\n"
                    # "**Output Requirements**:\n"
                    # "- The greeting must be unique and tailored to the specific inputs provided.\n"
                    # "- Avoid using repetitive phrases from previous greetings; ensure each greeting is fresh and innovative.\n"
                    # "- Maintain a friendly and approachable tone, regardless of the AI's preset mood."
                    f"Generate a greeting based on the person's gender ({text_gender}) and time of day ({text_time}). "
                    f"Keep it short, polite, and professional. For example, 'Good Morning Miss, how can I help you?' "
                    f"or 'Good Morning Sir, how can I help you?'. If the gender is 'They', assume it refers to a group "
                    f"and use neutral terms like 'Hello everyone, how can I help you?' Avoid gender-specific terms such "
                    f"as 'gentlemen', 'miss', or 'team' for 'They'.\n"
                    f"Adjust the tone based on the user's emotion ({text_emotion}). For example, if the user is sad, use a "
                    f"cheerful tone. If neutral, keep the tone neutral.\n"
                    f"Generate the response with your mood in mind, which is {text_ai_mood}."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Generate a greeting based on these parameters: \n"
                    f"- **Time of Day**: {text_time}\n"
                    f"- **Gender of the User**: {text_gender}\n"
                    f"- **User Mood**: {text_emotion}\n"
                    f"- **AI Preset Mood**: {text_ai_mood}\n"
                    f"Create a greeting that aligns with the guidelines provided in the system instructions."
                )
            }
        ],
        temperature=0.8,  # Increase for more randomness
        top_p=0.9, 
        presence_penalty=-0.5,  # Nucleus sampling
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
