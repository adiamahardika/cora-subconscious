from openai import OpenAI
import os

class LLMHandler:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API key must be provided either as an argument or through the OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=self.api_key)

    def send_prompt(self, system_prompt, user_prompt, model="gpt-4o-mini", temperature=0.8, top_p=0.9, presence_penalty=-0.5):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
            )

            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            return None