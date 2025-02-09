from openai import OpenAI
import azure.cognitiveservices.speech as speechsdk
import os
import io

class TTSHandler:
    def generate_speech_openai(self, text, voice = "alloy", model="tts-1"):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key must be provided either as an argument or through the OPENAI_API_KEY environment variable.")
        
        client = OpenAI(api_key=api_key)

        try:
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format="wav"
            )

            buffer = io.BytesIO()
            
            for chunk in response.iter_bytes(1024):
                buffer.write(chunk)
            buffer.seek(0)

            return buffer
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def generate_speech_azure(self, text):
        api_key = os.getenv('AZURE_SPEECH_KEY')
        service_region = "southeastasia"

        try:

            speech_config = speechsdk.SpeechConfig(subscription=api_key, region=service_region)
            speech_config.speech_synthesis_voice_name = "en-US-JennyMultilingualNeural"

            output_filename = "app/media/output_speech.wav"
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False, filename=output_filename)

            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

            result = speech_synthesizer.speak_text_async(text).get()

            # Check result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print("Speech synthesized for text [{}]".format(text))

                with open(output_filename, 'rb') as f:
                    audio_data = f.read()  # Read the file content into memory
                    audio_stream = io.BytesIO(audio_data) 
                
                return audio_stream 
            
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                print("Speech synthesis canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print("Error details: {}".format(cancellation_details.error_details))

        except Exception as e:
            print(f"An error occurred: {e}")
            return None