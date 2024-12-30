Important!

# Backend of AI Frontdesk Web Application
---------------------

# Table of Contents

1. Overview
2. Requirements
3. Installation
4. Setup
5. Usage
6. ai_speech API

---------------------

# Overview

This is the backend of the Frontdesk Web Application, the backend is developed using Flask and Python

---------------------

# Requirements

- Python Version: 3.12.5
- Environment Variable: Open AI API Key
- Required Libraries: Refer to requirements.txt

---------------------

# Installation

1. Clone the Repository using git clone command.

2. (Optional) Create a virtual environment in the project folder using python -m venv venv, activate virtual environment using ".\venv\Scripts\activate" for windows and "source venv/bin/activate" for mac.

3. Download all required libraries and modules through the requirements.txt file. 
   - Run the command "pip install -r requirements.txt".

---------------------

# Setup

1. Configure the .env file
   - The env file should be named .env and be placed in the ai_speech folder.
   - the env file should contain the environment value for Open AI Key.
   - the variable should be named "OPENAI_API_KEY"
   - As of 27/12/2024, the program uses ChatGPT model gpt-4o to accomplish the AI voice synthesize and text generation.

---------------------

# Usage

4. Run the Program
   - Run the command "python backend.py" to run the flask backend of the application.

---------------------

# ai_speech API

There are two API's under the ai_speech module

1. generate-greeting

returns a generated greeting from ChatGPT API in the form of a string, the parameters taken into account are:

User's Gender: text_gender
User's Current Mood: text_emotion
AI's Preset Mood: text_ai_mood
Time of Day: text_time

Parameters to control the output of the AI include the temperature, Top-p (Nucleus Sampling), and Presence Penalty

Generated text is saved to variable generated_text and sent to the front end in the json format.

2. generate-audio

returns a wav file to the front end, the parameters it uses are gender and text:

gender: the preset gender of the voice
text: text to be turned into speech.

Voice synthesis uses ChatGPT tts models for its TTS capabilities.

---------------------