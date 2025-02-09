from app.handler.llm_handler import LLMHandler

class LLMUseCase:
    def __init__(self):
        self.llm_handler = LLMHandler()

    def generate_greeting(self, gender, time, emotion, tone):
        system_prompt = (
                            f"Generate a greeting based on the person's gender ({gender}) and time of day ({time}). "
                            f"Match the tone and style to the specified guidelines:\n"
                            f"- If the gender is 'They,' assume it refers to a group and use neutral terms like 'Hello everyone' or 'Hi all.' "
                            f"Avoid gender-specific terms such as 'gentlemen,' 'miss,' or 'team' for 'They.'\n"
                            f"- Adjust the greeting to suit the user's emotion ({emotion}). For example, if the user is sad, use an empathetic or cheerful tone. "
                            f"If the user is neutral, maintain a neutral tone. If the user is happy, mirror their enthusiasm.\n"
                            f"- Use the specified style ({tone}) to determine the tone of the greeting:\n"
                            f"  1. Casual: Friendly and informal tone. Example: 'Hey there! Good morning! What's up?'\n"
                            f"  2. Santai: Relaxed but polite tone. Example: 'Hi, good morning! How can I help you today?'\n"
                            f"  3. Profesional: Polished and formal tone. Example: 'Good Morning. How may I assist you today?'\n"
                            f"Craft the response according to these guidelines."
                            f"Always answer in Bahasa Indonesia (Indonesian Language)"
                        )
        
        user_prompt = (
                            f"Generate a greeting based on these parameters:\n"
                            f"- **Time of Day**: {time}\n"
                            f"- **Gender of the User**: {gender}\n"
                            f"- **User Mood**: {emotion}\n"
                            f"- **AI Preset Mood**: {tone}\n"
                            f"Create a greeting that aligns with the guidelines provided in the system instructions."
                        )

        response = self.llm_handler.send_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        return response