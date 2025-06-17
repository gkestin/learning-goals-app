import os
from flask import Flask, request, redirect, url_for
from flask_cors import CORS
from config import Config
from datetime import datetime

def create_app(config_class=Config):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)

    # Configure CORS for custom domain
    CORS(app, origins=app.config['CORS_ORIGINS'])

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path, exist_ok=True)
        os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)
    except OSError:
        pass

    # HTTPS redirect for production custom domain
    @app.before_request
    def force_https():
        if app.config['FORCE_HTTPS'] and not request.is_secure:
            # Check if we're on the custom domain and not localhost
            if request.host and app.config['CUSTOM_DOMAIN'] in request.host:
                if not request.headers.get('X-Forwarded-Proto') == 'https':
                    return redirect(request.url.replace('http://', 'https://'), code=301)

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