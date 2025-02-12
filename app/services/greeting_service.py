from flask import jsonify, request, Response, send_file
from app.utils import response 
from app.usecases.llm_use_case import LLMUseCase
from app.usecases.tts_use_case import TTSUseCase
from marshmallow import Schema, fields, validate

def index():
    try:
        data = request.get_json()

        schema = Schema.from_dict({
            "gender": fields.String(required=True, validate=validate.OneOf(["Male", "Female"])),
            "time": fields.Time(required=True, format="%H:%M:%S"),
            "emotion": fields.String(required=True, validate=validate.OneOf(["happy", "sad", "angry", "neutral"])),
            "tone": fields.String(required=True, validate=validate.OneOf(["casual", "santai", "profesional"]))
        })()

        errors = schema.validate(data)

        if errors:
            return response(errors=errors), 400 

        gender = data.get('gender')
        time = data.get('time')
        emotion = data.get('emotion')
        tone = data.get('tone')

        # Asynchronous API call to OpenAI
        use_case = LLMUseCase()
        greeting = use_case.generate_greeting(gender=gender, time=time, emotion=emotion, tone=tone)
        
        return response(greeting, None, status="success", message="Greetings generated successfully")
    except Exception as e:
        return response([], None, status="error", message=f"An error occurred: {str(e)}")
    

def text_to_speech():
    try:
        data = request.get_json()

        schema = Schema.from_dict({
            "gender": fields.String(required=True, validate=validate.OneOf(["male", "female"])),
            "text": fields.String(required=True)
        })()

        errors = schema.validate(data)

        if errors:
            return response(errors=errors), 400 

        gender = data.get('gender')
        text = data.get('text')

        # Asynchronous API call to OpenAI
        use_case = TTSUseCase()
        speech = use_case.generate_speech(gender=gender, text=text)

        return Response(
            speech,
            content_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=audio.wav"}
        )
    except Exception as e:
        return response([], None, status="error", message=f"An error occurred: {str(e)}")


def conversation():
    try:
        data = request.get_json()

        schema = Schema.from_dict({
            "gender": fields.String(required=True, validate=validate.OneOf("Male", "Female")),
            "time": fields.Time(required=True, format="%H:%M:%S"),
            "emotion": fields.String(required=True, validate=validate.OneOf(["happy", "sad", "angry", "neutral"])),
            "tone": fields.String(required=True, validate=validate.OneOf(["casual", "santai", "profesional"]))
        })()

        errors = schema.validate(data)

        if errors:
            return response(errors=errors), 400 

        gender = data.get('gender')
        time = data.get('time')
        emotion = data.get('emotion')
        tone = data.get('tone')

        llm_use_case = LLMUseCase()
        tts_use_case = TTSUseCase()

        greeting = llm_use_case.generate_greeting(gender=gender, time=time, emotion=emotion, tone=tone)
        speech = tts_use_case.generate_speech(gender=gender, text=greeting)

        return Response(
            speech,
            content_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=audio.wav"}
        )
    except Exception as e:
        return response([], None, status="error", message=f"An error occurred: {str(e)}")
