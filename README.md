# Learning Goals Extractor

An AI-powered Flask application that extracts learning goals from PDF documents using OpenAI's API and stores both the documents and extracted data in Firebase.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Local Development Setup](#local-development-setup)
- [Firebase Setup](#firebase-setup)
- [Google Cloud Setup](#google-cloud-setup)
- [Running Locally](#running-locally)
- [Ultra-Fast Deployment](#ultra-fast-deployment)
- [Git Version Control](#git-version-control)
- [Essential Commands](#essential-commands)
- [Troubleshooting](#troubleshooting)

## Features

- PDF upload and text extraction using PyPDF2 and pdfplumber
- AI-powered learning goal extraction using OpenAI's GPT models
- Firebase Storage for PDF file storage
- Firebase Firestore for metadata and extracted data storage
- Web interface for uploading PDFs and viewing extracted goals
- Google Cloud Run deployment for scalable hosting

## Project Structure

```
.
├── main.py                    # Flask application entry point
├── app.py                     # Main application logic
├── firebase-key.json         # Firebase service account key (keep private)
├── requirements.txt           # Python dependencies
├── Dockerfile.app            # Lightweight app Docker image
├── docker/
│   └── base.Dockerfile       # Heavy ML dependencies base image
├── deploy-base.sh            # Build base image (run when requirements.txt changes)
├── deploy-ultra-fast.sh      # Ultra-fast deployment (~3 minutes)
├── deploy.sh                 # Legacy slow deployment (20+ min, avoid)
├── .dockerignore             # Docker build optimization
├── templates/                # HTML templates
└── static/                   # Static CSS/JS files
```

## Local Development Setup

1. **Clone the repository** and navigate to the project directory

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** in `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   SECRET_KEY=your_secret_key_here
   FIREBASE_STORAGE_BUCKET=your_project.firebasestorage.app
   ```

## Firebase Setup

1. **Create a Firebase project** at https://console.firebase.google.com/
2. **Enable Firestore Database** and **Storage**
3. **Create a service account**:
   - Go to Project Settings → Service Accounts
   - Generate a new private key
   - Save as `firebase-key.json` in project root
4. **Set up security rules** for Firestore and Storage (allow read/write for development)

## Google Cloud Setup

1. **Install Google Cloud CLI**: https://cloud.google.com/sdk/docs/install
2. **Authenticate**: `gcloud auth login`
3. **Set project**: `gcloud config set project YOUR_PROJECT_ID`
4. **Enable APIs**:
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

## Running Locally

```bash
python main.py
```
Visit http://localhost:8080 to access the application.

## Ultra-Fast Deployment

### **🚀 Two-Script Deployment System**

This project uses a **two-stage deployment** approach for maximum speed:

#### **1. Base Image Setup (Rare)**
Run when you change `requirements.txt` or first-time setup:
```bash
./deploy-base.sh
```
- **Duration**: ~10 minutes
- **Contains**: All heavy ML dependencies (PyTorch, etc.)
- **Frequency**: Only when adding/changing Python packages

#### **2. Ultra-Fast App Deployment (Regular)**
Run for every code change:
```bash
./deploy-ultra-fast.sh
```
- **Duration**: ~3 minutes
- **Contains**: Just your application code
- **Frequency**: Every time you update your app

### **Why This Is Fast**
- **Local Docker Cache**: Base image (1GB+) stays cached on your machine
- **Layer Reuse**: Only changed code gets rebuilt/pushed
- **No Cloud Build**: Skips slow ephemeral build workers
- **7x Speed Improvement**: From 20+ minutes to ~3 minutes

### **Deployment Configuration**
Create `deploy-config.sh` with your OpenAI API key:
```bash
#!/bin/bash
export DEPLOY_OPENAI_API_KEY="your_openai_api_key_here"
```

### **First Time Setup**
1. Run `./deploy-base.sh` (builds base image with ML dependencies)
2. Run `./deploy-ultra-fast.sh` (deploys your app)
3. Future deployments: just `./deploy-ultra-fast.sh`

### **When to Use Each Script**
- **`./deploy-base.sh`**: Changed requirements.txt, added Python packages
- **`./deploy-ultra-fast.sh`**: Changed application code, templates, static files

## Git Version Control

### Essential Git Commands
```bash
# Check status
git status

# Add files
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push origin main

# Pull latest changes
git pull origin main
```

## Essential Commands

### Local Development
```bash
# Start development server
python main.py

# Install new package
pip install package_name
pip freeze > requirements.txt  # Update requirements after installing packages
```

### Deployment
```bash
# One-time or when requirements change
./deploy-base.sh

# Regular deployment (every code change)
./deploy-ultra-fast.sh

# Check deployment status
gcloud run services list
```

### Docker Management
```bash
# View Docker images
docker images

# Clean up old images (if disk space is low)
docker system prune
```

## Troubleshooting

### Common Issues

1. **"Base image not found"**
   - Run `./deploy-base.sh` first to build the base image

2. **Docker authentication errors**
   - Run `gcloud auth configure-docker`

3. **OpenAI API errors**
   - Check your API key in `deploy-config.sh`
   - Verify you have sufficient API credits

4. **Firebase errors**
   - Ensure `firebase-key.json` exists and has correct permissions
   - Check Firebase project settings

5. **Slow deployment**
   - Use `./deploy-ultra-fast.sh` instead of old deployment scripts
   - Ensure base image is built with `./deploy-base.sh`

### Performance Monitoring
- **Cloud Run logs**: `gcloud logs read --service learning-goals-app`
- **Application URL**: Check deployment output for service URL

### Getting Help
- Check Cloud Run console: https://console.cloud.google.com/run
- View build logs in Cloud Build console
- Monitor Firebase usage in Firebase console
