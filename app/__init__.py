import os
from flask import Flask
from config import Config
from datetime import datetime

def create_app(config_class=Config):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path, exist_ok=True)
        os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)
    except OSError:
        pass

    # Initialize services
    from app import firebase_service
    firebase_service.init_app(app)

    # Add context processor for templates
    @app.context_processor
    def inject_now():
        return {'now': datetime.now()}

    # Register blueprints
    from app.routes import main
    app.register_blueprint(main)

    return app 