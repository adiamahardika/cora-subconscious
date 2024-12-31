from flask import Blueprint

ai_speech_bp = Blueprint('ai_speech', __name__)

from .routes import ai_greeting  # Import your custom route to register it