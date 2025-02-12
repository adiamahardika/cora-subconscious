from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os
from .object_detection import start_inference

db = SQLAlchemy()
migrate = Migrate()

load_dotenv()

socketio = SocketIO()


def create_app():
    app = Flask(__name__)

    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
    app.config['FLASK_ENV'] = os.getenv('FLASK_ENV')
    
    CORS(app)
    
    socketio.init_app(app, cors_allowed_origins="*", async_mode = 'threading')

    from .events import register_events
    register_events()

    db.init_app(app)
    migrate.init_app(app, db)

    with app.app_context():
        from app.routes import bp
        app.register_blueprint(bp)
        socketio.run(app, debug=True, host="0.0.0.0", port=5000)
        socketio.start_background_task(start_inference)

    return app