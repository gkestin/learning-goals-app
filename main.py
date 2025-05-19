from app import create_app
import os
import socket
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Print environment variables for debugging
desired_port = os.environ.get('PORT', '3000')
print(f"\n==== ENVIRONMENT VARIABLES ====")
print(f"PORT: {desired_port}")
print(f"FIREBASE_STORAGE_BUCKET: {os.environ.get('FIREBASE_STORAGE_BUCKET', 'Not set')}")
print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', 'Not set')}")
print(f"Working directory: {os.getcwd()}")
print(f"==== END ENV VARS ====\n")

def find_available_port(start_port=3000, max_attempts=10):
    """Find an available port starting from start_port"""
    port = int(start_port)
    for attempt in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('0.0.0.0', port))
                print(f"Found available port: {port}")
                return port
        except OSError:
            port += 1
            print(f"Port {port-1} is in use, trying {port}...")
    
    # If we get here, we couldn't find an available port
    print(f"Couldn't find an available port after {max_attempts} attempts")
    return int(start_port) + max_attempts

# Get desired port from env or default to 3000
desired_port = os.environ.get('PORT', '3000')
print(f"Desired port from environment: {desired_port}")

app = create_app()

if __name__ == '__main__':
    # Find an available port
    port = find_available_port(start_port=desired_port)
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 