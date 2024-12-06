from langchain_google_genai import ChatGoogleGenerativeAI
from flask import request, jsonify
import os
from flask_socketio import SocketIO
from .. import ai_speech_bp
# Get the absolute path to the config file
# Get the directory of the current file
current_dir = os.path.dirname(__file__)

config_dir = os.path.abspath(os.path.join(os.path.dirname(
    __file__), "../config/tts-frontdesk-79a0044d0419.json"))
print(f"Config file path: {config_dir}")

if not os.path.isfile(config_dir):
    raise Exception(f"Credentials file not found at {config_dir}")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config_dir
llm = ChatGoogleGenerativeAI(model="gemini-pro")

@ai_speech_bp.route('/generate-greeting', methods=['POST'])
def synthesize_greeting():
    if request.is_json:
        text_block = request.json.get('text')
    else:
        return jsonify({'error': 'Invalid content type, expected application/json'}), 400

    if not text_block:
        return jsonify({'error': 'Text is required'}), 400

    prompt = "Generate a Greeting based on the person's gender, Gender: "
    response = llm.invoke(prompt + text_block)

    generated_text = response.content  # Get the response text from LLM
    print(response.content)

    if not generated_text:
        return jsonify({'error': 'No response generated from LLM'}), 400
    
    

    # Return the generated text to the frontend
    return jsonify({"text": generated_text})

# Route to trigger ping via a button


