from app.handler.tts_handler import TTSHandler

class TTSUseCase:
    def __init__(self):
        self.tts_handler = TTSHandler()

    def generate_speech(self, gender, text):
        voice = "onyx" if gender == "male" else "nova" if gender == "female" else "alloy"

        # response = self.tts_handler.generate_speech_openai(text, voice)
        response = self.tts_handler.generate_speech_azure(text)

        return response