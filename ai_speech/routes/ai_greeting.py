from flask import request, jsonify, Response
import os
from .. import ai_speech_bp
from dotenv import load_dotenv
from openai import OpenAI
import datetime
import io

# Get the absolute path to the config file
# Get the directory of the current file

env_path = os.path.join(os.path.dirname(__file__), "../", ".env")

current_dir = os.path.dirname(__file__)

load_dotenv(dotenv_path=env_path)

openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_api_key)


@ai_speech_bp.route('/generate-greeting', methods=['POST'])
def synthesize_greeting():
    # Validate request content type
    if not request.is_json:
        return jsonify({'error': 'Invalid content type, expected application/json'}), 400

    # Extract request parameters
    data = request.get_json()
    text_gender = data.get('user_gender')
    text_time = data.get('time')
    text_emotion = data.get('emotion')
    text_ai_mood = data.get('tone')

    # Validate required fields
    if not text_gender:
        return jsonify({'error': 'Text (gender) is required'}), 400

    if not text_time or not text_emotion or not text_ai_mood:
        return jsonify({'error': 'All fields (time, emotion, tone) are required'}), 400

    try:
        # Debugging logs
        print(f"Parameters received: Gender: {text_gender}, Time: {text_time}, Emotion: {text_emotion}, Tone: {text_ai_mood}")

        # Asynchronous API call to OpenAI
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Generate a greeting based on the person's gender ({text_gender}) and time of day ({text_time}). "
                            f"Match the tone and style to the specified guidelines:\n"
                            f"- If the gender is 'They,' assume it refers to a group and use neutral terms like 'Hello everyone' or 'Hi all.' "
                            f"Avoid gender-specific terms such as 'gentlemen,' 'miss,' or 'team' for 'They.'\n"
                            f"- Adjust the greeting to suit the user's emotion ({text_emotion}). For example, if the user is sad, use an empathetic or cheerful tone. "
                            f"If the user is neutral, maintain a neutral tone. If the user is happy, mirror their enthusiasm.\n"
                            f"- Use the specified style ({text_ai_mood}) to determine the tone of the greeting:\n"
                            f"  1. Casual: Friendly and informal tone. Example: 'Hey there! Good morning! What's up? 😊'\n"
                            f"  2. Santai: Relaxed but polite tone. Example: 'Hi, good morning! How can I help you today?'\n"
                            f"  3. Profesional: Polished and formal tone. Example: 'Good Morning. How may I assist you today?'\n"
                            f"Craft the response according to these guidelines."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Generate a greeting based on these parameters:\n"
                            f"- **Time of Day**: {text_time}\n"
                            f"- **Gender of the User**: {text_gender}\n"
                            f"- **User Mood**: {text_emotion}\n"
                            f"- **AI Preset Mood**: {text_ai_mood}\n"
                            f"Create a greeting that aligns with the guidelines provided in the system instructions."
                        )
                    }
                ],
                temperature=0.8,
                top_p=0.9,
                presence_penalty=-0.5,
            )
        
        choices = completion.choices
        chat_completion = choices[0]
        # Extract and return the generated text
        generated_text = chat_completion.message.content
        
        print(generated_text)
        
        return jsonify({"text": generated_text})


    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': f'Failed to generate greeting: {str(e)}'}), 500
    

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
        
        try:
            for chunk in response.iter_bytes(1024):
                buffer.write(chunk)
            buffer.seek(0)
            return Response(
                buffer,
                content_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=audio.wav"}
            )
        except Exception as e:
            print(f"Error generating audio: {e}")
            return jsonify({"error": str(e)}), 500
    
    except Exception as e:
        print(f"Error generating audio: {e}")
        return jsonify({"error": str(e)}), 500
