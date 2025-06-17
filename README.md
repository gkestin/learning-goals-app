# Learning Goals Extractor

🚀 **Live at: [https://mathmatic.org](https://mathmatic.org)**

An AI-powered Flask application that extracts learning goals from PDF documents using OpenAI's API and stores both the documents and extracted data in Firebase. Now featuring advanced clustering analysis and hierarchical goal organization.

## 🌟 New Features
- **🌐 Live Production Site**: Deployed at [mathmatic.org](https://mathmatic.org)
- **🧠 Advanced Clustering**: HDBSCAN clustering with interactive visualizations
- **🌳 Hierarchical Analysis**: Tree-based goal organization and navigation
- **📊 Interactive Charts**: Plotly-powered cluster visualizations
- **🔍 Smart Search**: Search and filter through goal hierarchies
- **💾 Artifact Management**: Save and manage clustering results
- **🎯 Custom Domain**: Professional deployment with SSL

## Table of Contents
- [🌟 New Features](#-new-features)
- [🚀 Live Application](#-live-application)
- [📋 Features](#features)
- [📁 Project Structure](#project-structure)
- [🛠️ Local Development Setup](#️-local-development-setup)
- [🔥 Firebase Setup](#-firebase-setup)
- [☁️ Google Cloud Setup](#️-google-cloud-setup)
- [🏃 Running Locally](#-running-locally)
- [⚡ Ultra-Fast Deployment](#-ultra-fast-deployment)
- [🌐 Custom Domain Deployment](#-custom-domain-deployment)
- [📝 Git Version Control](#-git-version-control)
- [🔧 Essential Commands](#-essential-commands)
- [🆘 Troubleshooting](#-troubleshooting)

## 🚀 Live Application

**Production URL**: [https://mathmatic.org](https://mathmatic.org)
- Fully deployed on Google Cloud Run
- Custom domain with SSL certificate
- Auto-scaling and high availability

## 📋 Features

### Core Features
- PDF upload and text extraction using PyPDF2 and pdfplumber
- AI-powered learning goal extraction using OpenAI's GPT models
- Firebase Storage for PDF file storage
- Firebase Firestore for metadata and extracted data storage
- Web interface for uploading PDFs and viewing extracted goals

### Advanced Analytics
- **HDBSCAN Clustering**: Advanced machine learning clustering
- **Interactive Visualizations**: Plotly charts and graphs
- **Hierarchical Tree Views**: Navigate goal relationships
- **Smart Search & Filter**: Find specific goals and clusters
- **Artifact Management**: Save and reload analysis results
- **Export Capabilities**: Download results in multiple formats

### Deployment Features
- Google Cloud Run deployment for scalable hosting
- Custom domain support (mathmatic.org)
- HTTPS redirect and security headers
- CORS configuration for cross-origin requests
- Auto-scaling based on traffic

## 📁 Project Structure

```
.
├── main.py                           # Flask application entry point
├── app/
│   ├── __init__.py                   # App initialization
│   ├── routes.py                     # Application routes and views
│   ├── models.py                     # Data models
│   ├── clustering_service.py         # HDBSCAN clustering logic
│   ├── firebase_service.py           # Firebase integration
│   ├── openai_service.py             # OpenAI API integration
│   ├── pdf_utils.py                  # PDF processing utilities
│   ├── templates/                    # HTML templates
│   │   ├── base.html                 # Base template
│   │   ├── index.html                # Main upload page
│   │   ├── view.html                 # Goal viewing page
│   │   ├── cluster.html              # Clustering interface
│   │   ├── cluster_tree.html         # Tree visualization
│   │   ├── artifacts.html            # Artifact management
│   │   └── search.html               # Search interface
│   └── static/
│       └── lock-manager.js           # Frontend state management
├── config.py                        # Application configuration
├── firebase-key.json                # Firebase service account key (keep private)
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Production Docker image
├── Dockerfile.app                    # Lightweight app Docker image
├── docker/
│   └── base.Dockerfile              # Heavy ML dependencies base image
├── deploy-mathmatic.sh              # 🆕 One-click MathMatic.org deployment
├── deploy-base.sh                   # Build base image (run when requirements.txt changes)
├── deploy-ultra-fast.sh             # Ultra-fast deployment (~3 minutes)
├── deploy.sh                        # Full deployment with domain setup
├── setup-domain.sh                  # Domain configuration
├── complete-domain-setup.sh         # Final domain mapping
├── DEPLOYMENT_GUIDE_MATHMATIC.md    # 🆕 Complete deployment guide
├── .dockerignore                    # Docker build optimization
└── sample_learning_goals_hierarchy.csv  # Sample data
```

## 🛠️ Local Development Setup

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
   CUSTOM_DOMAIN=localhost
   FORCE_HTTPS=false
   ```

## 🔥 Firebase Setup

1. **Create a Firebase project** at https://console.firebase.google.com/
2. **Enable Firestore Database** and **Storage**
3. **Create a service account**:
   - Go to Project Settings → Service Accounts
   - Generate a new private key
   - Save as `firebase-key.json` in project root
4. **Set up security rules** for Firestore and Storage (allow read/write for development)

## ☁️ Google Cloud Setup

1. **Install Google Cloud CLI**: https://cloud.google.com/sdk/docs/install
2. **Authenticate**: `gcloud auth login`
3. **Set project**: `gcloud config set project YOUR_PROJECT_ID`
4. **Enable APIs**:
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

## 🏃 Running Locally

```bash
python main.py
```
Visit http://localhost:3000 to access the application.

## ⚡ Ultra-Fast Deployment

### **🚀 Two-Script Deployment System**

This project uses a **two-stage deployment** approach for maximum speed:

#### **1. Base Image Setup (Rare)**
Run when you change `requirements.txt` or first-time setup:
```bash
./deploy-base.sh
```
- **Duration**: ~10 minutes
- **Contains**: All heavy ML dependencies (PyTorch, scikit-learn, etc.)
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

## 🌐 Custom Domain Deployment

### **🎯 One-Click MathMatic.org Deployment**

For easy deployment to the production domain:

```bash
./deploy-mathmatic.sh
```

This interactive script will:
1. ✅ Check prerequisites (gcloud, Docker, API keys)
2. 🚀 Deploy your app to Google Cloud Run
3. 🌐 Set up domain mappings for mathmatic.org
4. 📋 Guide you through DNS configuration

### **Manual Domain Setup**

For detailed control or troubleshooting:

1. **Deploy the app**:
   ```bash
   ./deploy-ultra-fast.sh
   ```

2. **Set up domain**:
   ```bash
   ./setup-domain.sh
   ```

3. **Complete domain verification**:
   ```bash
   ./complete-domain-setup.sh
   ```

### **Domain Configuration Requirements**

**DNS Records** (add to your domain provider):
```dns
Type: A, Host: @, Value: 216.239.32.21
Type: A, Host: @, Value: 216.239.34.21  
Type: A, Host: @, Value: 216.239.36.21
Type: A, Host: @, Value: 216.239.38.21
Type: CNAME, Host: www, Value: ghs.googlehosted.com.
```

**Domain Verification**: Required through Google Search Console

### **Deployment Configuration**
Create `deploy-config.sh` with your OpenAI API key:
```bash
#!/bin/bash
export DEPLOY_OPENAI_API_KEY="your_openai_api_key_here"
```

### **When to Use Each Script**
- **`./deploy-mathmatic.sh`**: 🎯 One-click production deployment
- **`./deploy-base.sh`**: 🔧 Changed requirements.txt, added Python packages
- **`./deploy-ultra-fast.sh`**: ⚡ Changed application code, templates, static files
- **`./deploy.sh`**: 🌐 Full deployment with domain setup
- **`./setup-domain.sh`**: 🔗 Domain configuration only

## 📝 Git Version Control

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

## 🔧 Essential Commands

### Local Development
```bash
# Start development server
python main.py

# Install new package
pip install package_name
pip freeze > requirements.txt  # Update requirements after installing packages
```

### Production Deployment
```bash
# 🎯 One-click production deployment
./deploy-mathmatic.sh

# ⚡ Quick updates (most common)
./deploy-ultra-fast.sh

# 🔧 When requirements change
./deploy-base.sh

# 🌐 Domain setup only
./setup-domain.sh

# Check deployment status
gcloud run services list
```

### Monitoring
```bash
# View application logs
gcloud logs read --service learning-goals-app

# Check domain mappings
gcloud beta run domain-mappings list --region us-central1

# Monitor resource usage
gcloud run services describe learning-goals-app --region us-central1
```

## 🆘 Troubleshooting

### Common Issues

1. **🚫 Domain not working**
   - Check DNS propagation: https://dnschecker.org
   - Verify domain ownership in Google Search Console
   - Run `./complete-domain-setup.sh` after verification

2. **🔒 SSL certificate issues**
   - Wait 10-15 minutes for automatic provisioning
   - Ensure DNS records are correct
   - Check domain mapping status

3. **🐳 "Base image not found"**
   - Run `./deploy-base.sh` first to build the base image

4. **🔐 Docker authentication errors**
   - Run `gcloud auth configure-docker`

5. **🤖 OpenAI API errors**
   - Check your API key in `deploy-config.sh`
   - Verify you have sufficient API credits

6. **🔥 Firebase errors**
   - Ensure `firebase-key.json` exists and has correct permissions
   - Check Firebase project settings

7. **🐌 Slow deployment**
   - Use `./deploy-ultra-fast.sh` instead of old deployment scripts
   - Ensure base image is built with `./deploy-base.sh`

### Performance Monitoring
- **🌐 Production URL**: [https://mathmatic.org](https://mathmatic.org)
- **☁️ Cloud Run Console**: https://console.cloud.google.com/run
- **🔥 Firebase Console**: https://console.firebase.google.com
- **📊 Application Logs**: `gcloud logs read --service learning-goals-app`

### 📚 Documentation
- **📖 Deployment Guide**: `DEPLOYMENT_GUIDE_MATHMATIC.md`
- **🌐 Custom Domain Setup**: Detailed steps in deployment guide
- **🔧 Troubleshooting**: Common solutions for domain and SSL issues

### Getting Help
- Check Cloud Run console: https://console.cloud.google.com/run
- View build logs in Cloud Build console
- Monitor Firebase usage in Firebase console
- Test direct Cloud Run URL if domain issues persist
