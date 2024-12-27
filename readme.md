Important!

Python Version: 3.12.5

1. Download all required libraries and modules through the requirements.txt file. 
   - (Optional) Create a virtual environment in the project folder using python -m venv venv, activate afterwards by navigating to the Scripts folder and using the activate script.
   - Run the command "pip install -r requirements.txt".

2. Setup the .env file
   - The env file should be named .env and be placed in the ai_speech folder.
   - the env file should contain the environment value for Open AI Key.
   - the variable should be named "OPENAI_API_KEY"
   - As of 27/12/2024, the program uses ChatGPT model gpt-4o to accomplish the AI voice synthesize and text generation.

3. Run the Program
   - Run the command "python backend.py" to run the flask backend of the application.