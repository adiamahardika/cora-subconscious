
# How to Run

## Prerequisites

- Python 3.9
- Ensure all required libraries are installed (refer to `requirements.txt`)

## Installation

1. **Clone the Repository**

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Create and Activate Virtual Environment (Optional but recommended)**

   - For Windows:
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - For macOS/Linux:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```

3. **Install Required Libraries**

   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. **Configure the .env File**

   - Create a `.env` file in the root directory of the project.
   - Add the following environment variable:
     ```plaintext
     OPENAI_API_KEY=your_openai_api_key
     ```

## Running the Application

1. **Set the Flask App Environment Variable**

   - For Windows:
     ```bash
     set FLASK_APP=app
     set FLASK_ENV=development
     ```
   - For macOS/Linux:
     ```bash
     export FLASK_APP=app
     export FLASK_ENV=development
     ```

2. **Run the Flask Application**

   ```bash
   flask run
   ```

   The application will be available at `http://127.0.0.1:5000/`.

## Additional Commands
- **Initialize the Database**

  ```bash
  flask db init
  ```

- **Create Database Migrations**

  ```bash
  flask db migrate -m "migration message"
  ```

- **Run Database Migrations**

  ```bash
  flask db upgrade
  ```


## Testing

- **Run Tests**

  ```bash
  pytest
  ```

## AI Speech API

This module uses the ChatGPT model `gpt-4o` for AI voice synthesis and text generation.

There are two APIs under the `ai_speech` module:

### 1. Generate Greeting

Returns a generated greeting from the ChatGPT API in the form of a string. The parameters taken into account are:

- **User's Gender:** `text_gender`
- **User's Current Mood:** `text_emotion`
- **AI's Preset Mood:** `text_ai_mood`
- **Time of Day:** `text_time`

Parameters to control the output of the AI include the temperature, Top-p (Nucleus Sampling), and Presence Penalty.

The generated text is saved to the variable `generated_text` and sent to the front end in JSON format.

### 2. Generate Audio

Returns a WAV file to the front end. The parameters it uses are:

- **Gender:** The preset gender of the voice
- **Text:** Text to be turned into speech

Voice synthesis uses ChatGPT `Text to Speech` models for its capabilities.
