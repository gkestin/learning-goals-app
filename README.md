# Learning Goals Extractor

This is a Flask application that allows users to upload PDF documents, extract learning goals using the OpenAI API, and store both the document and learning goals in Firebase.

## Features

- PDF uploading and text extraction
- Learning goals extraction using OpenAI's GPT models
- Interactive editing of learning goals
- Document metadata management
- Firebase Firestore integration for data storage
- Google Cloud Storage integration for PDF storage
- Search functionality with autocomplete for learning goals
- Responsive UI built with Bootstrap

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Google Firebase project with Firestore and Cloud Storage
- Google Cloud CLI (optional, for deployment)

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd learning-goals-app
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Firebase project

1. Go to the [Firebase Console](https://console.firebase.google.com/)
2. Click "Add project" and follow the steps to create a new project
3. Enable Firestore Database:
   - Go to "Firestore Database" in the left sidebar
   - Click "Create database"
   - Choose "Start in production mode" and select a location

4. Enable Storage:
   - Go to "Storage" in the left sidebar
   - Click "Get started"
   - Accept the default security rules for now

5. Generate Firebase credentials:
   - Go to Project settings > Service accounts
   - Click "Generate new private key" for Firebase Admin SDK
   - Save the JSON file securely

### 5. Set up environment variables

1. Copy the example environment file:
   ```bash
   cp env.example .env
   ```

2. Edit the `.env` file with your specific credentials:
   - Add your OpenAI API key
   - Add the path to your Firebase credentials JSON file
   - Add your Firebase storage bucket name (found in Firebase Console > Storage)
   - Set a secret key for Flask

### 6. Run the application locally

```bash
python main.py
```

The application will be available at http://localhost:8080

## Deployment to Google Cloud Run

### 1. Install Google Cloud CLI

Follow the [official documentation](https://cloud.google.com/sdk/docs/install) to install the gcloud CLI.

### 2. Authenticate and configure Google Cloud

```bash
gcloud auth login
gcloud config set project YOUR_FIREBASE_PROJECT_ID
```

### 3. Build and deploy to Cloud Run

```bash
gcloud builds submit --tag gcr.io/YOUR_FIREBASE_PROJECT_ID/learning-goals-app
gcloud run deploy learning-goals-app --image gcr.io/YOUR_FIREBASE_PROJECT_ID/learning-goals-app --platform managed --allow-unauthenticated
```

### 4. Set environment variables in Cloud Run

In the Google Cloud Console:
1. Go to Cloud Run > your service
2. Go to "Edit & Deploy New Revision"
3. Add all the environment variables from your .env file
4. For the credentials file, you'll need to:
   - Create a Cloud Storage bucket
   - Upload your Firebase credentials JSON
   - Mount it as a volume in your Cloud Run service or use Google Secret Manager

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [OpenAI](https://openai.com/)
- [Firebase](https://firebase.google.com/)
- [Bootstrap](https://getbootstrap.com/)

## Firebase Configuration

When setting up Firebase Storage, make sure to use the correct bucket name format:

1. In the Firebase console, you'll see the bucket URL as `gs://your-project-id.firebasestorage.app`
2. In your `.env` file, set the `FIREBASE_STORAGE_BUCKET` variable to `your-project-id.firebasestorage.app` (without the `gs://` prefix)

**Important**: Don't use the `.appspot.com` domain format for Firebase Storage. The correct format is `.firebasestorage.app`. 