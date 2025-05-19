# Learning Goals Extractor

An AI-powered Flask application that extracts learning goals from PDF documents using OpenAI's API and stores both the documents and extracted data in Firebase.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Local Development Setup](#local-development-setup)
- [Firebase Setup](#firebase-setup)
- [Google Cloud Setup](#google-cloud-setup)
- [Running Locally](#running-locally)
- [Deployment](#deployment)
- [Git Version Control](#git-version-control)
- [Essential Commands](#essential-commands)
- [Troubleshooting](#troubleshooting)

## Features

- PDF uploading and text extraction
- Learning goals extraction using OpenAI's GPT models with custom system prompts
- Interactive editing of learning goals
- Document metadata management (name, creator, institution, type, notes)
- Firebase Firestore integration for data storage
- Google Cloud Storage integration for PDF storage
- Search functionality with autocomplete for learning goals
- Document deletion capability
- Responsive UI built with Bootstrap

## Project Structure

- `app/`: Main application code
  - `__init__.py`: Flask application initialization
  - `models.py`: Data models for documents and learning goals
  - `routes.py`: API endpoints and page routes
  - `openai_service.py`: OpenAI API integration
  - `firebase_service.py`: Firebase integration
  - `pdf_utils.py`: PDF text extraction utilities
  - `templates/`: HTML templates for the web interface
- `config.py`: Application configuration
- `main.py`: Application entry point
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container configuration
- `deploy.sh`: Deployment script for Google Cloud Run
- `deploy-config.sh`: Secret configuration for deployment (not in Git)

## Local Development Setup

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Google Firebase project with Firestore and Cloud Storage
- Google Cloud CLI

### 1. Clone the repository

```bash
git clone https://github.com/gkestin/learning-goals-app.git
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

### 4. Set up environment variables

Create a `.env` file in the project root:

```
FLASK_APP=main.py
FLASK_ENV=development
SECRET_KEY=your-secret-key
OPENAI_API_KEY=your-openai-api-key
FIREBASE_CREDENTIALS_PATH=path/to/your/firebase-key.json
FIREBASE_STORAGE_BUCKET=your-project-id.firebasestorage.app
```

## Firebase Setup

### 1. Create a Firebase project

1. Go to the [Firebase Console](https://console.firebase.google.com/)
2. Click "Add project" and follow the steps to create a new project
3. Note your project ID (e.g., "learninggoals2")

### 2. Enable Firestore Database

1. Go to "Firestore Database" in the left sidebar
2. Click "Create database"
3. Choose "Start in production mode" and select a location

### 3. Enable Storage

1. Go to "Storage" in the left sidebar
2. Click "Get started"
3. Accept the default security rules for now

### 4. Generate Firebase credentials

1. Go to Project settings > Service accounts
2. Click "Generate new private key" for Firebase Admin SDK
3. Save the JSON file as `firebase-key.json` in your project directory
4. Add this file to `.gitignore` to keep it secure

### 5. Note your storage bucket name

The storage bucket name follows the format: `your-project-id.firebasestorage.app`

## Google Cloud Setup

### 1. Set up Google Cloud CLI

```bash
# Install Google Cloud CLI (if not already installed)
# See: https://cloud.google.com/sdk/docs/install

# Login to Google Cloud
gcloud auth login

# Set the project
gcloud config set project learninggoals2
```

### 2. Create a service account for deployment

```bash
# Create service account
gcloud iam service-accounts create learning-goals-app --display-name="Learning Goals App Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding learninggoals2 \
  --member="serviceAccount:learning-goals-app@learninggoals2.iam.gserviceaccount.com" \
  --role="roles/firebase.admin"

gcloud projects add-iam-policy-binding learninggoals2 \
  --member="serviceAccount:learning-goals-app@learninggoals2.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## Running Locally

1. Make sure your virtual environment is activated
2. Set up your `.env` file with the required environment variables
3. Run the application:

```bash
python main.py
```

The application will be available at http://localhost:8080 by default.

## Deployment

### 1. Create deploy-config.sh

Create a file named `deploy-config.sh` to store sensitive information (this file is excluded from Git):

```bash
#!/bin/bash

# This file contains sensitive configuration and should not be committed to Git
# It is automatically loaded by deploy.sh

# OpenAI API Key
export DEPLOY_OPENAI_API_KEY="your-openai-api-key"

# Other sensitive configuration can be added here
```

Make it executable:

```bash
chmod +x deploy-config.sh
```

### 2. Run the deployment script

```bash
./deploy.sh
```

This script will:
1. Build a Docker container
2. Push it to Google Container Registry
3. Set up secrets in Google Secret Manager
4. Deploy the application to Google Cloud Run
5. Output the URL of your deployed application

## Git Version Control

### Initial Setup

This project is already set up with Git. If you're starting from scratch:

```bash
git init
git add .
git commit -m "Initial commit"
```

### Adding a Remote Repository

```bash
git remote add origin https://github.com/yourusername/your-repo-name.git
```

### Working with Git

```bash
# Check status
git status

# Add changes
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push origin main
```

### Git Best Practices

1. Never commit sensitive information (API keys, credentials)
2. Use the provided `.gitignore` file to exclude sensitive files
3. Store secrets in `deploy-config.sh` which is excluded from Git

## Essential Commands

### Local Development

```bash
# Start virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application locally
python main.py

# Run with a specific port
PORT=8000 python main.py
```

### Deployment

```bash
# Deploy to Google Cloud Run
./deploy.sh

# View logs of deployed application
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=learning-goals-app" --limit=50

# View the deployed application URL
gcloud run services describe learning-goals-app --platform managed --region us-central1 --format 'value(status.url)'
```

### Git Commands

```bash
# Check status
git status

# Add and commit changes
git add .
git commit -m "Description of changes"

# Push to GitHub
git push origin main

# Pull latest changes
git pull origin main
```

## Troubleshooting

### Firebase Connection Issues

- Verify your Firebase credentials file is properly formatted
- Check that your service account has the necessary permissions
- Ensure the storage bucket name is correctly formatted as `your-project-id.firebasestorage.app`

### Deployment Issues

- Check the logs: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=learning-goals-app" --limit=50`
- Verify that the service account has the right permissions
- Ensure all environment variables are properly set in the deployment

### OpenAI API Issues

- Check that your API key is valid
- Verify the system message formatting in `openai_service.py`
- Check for any changes in OpenAI's API that might require updates to our code

### Local Development Issues

- Make sure all environment variables are set in your `.env` file
- Verify that your virtual environment is activated
- Check that all dependencies are installed: `pip install -r requirements.txt`

### Accessing the App

The deployed application is available at:
https://learning-goals-app-375966757517.us-central1.run.app 