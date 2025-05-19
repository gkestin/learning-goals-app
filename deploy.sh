#!/bin/bash
set -e

# Configuration
PROJECT_ID="learninggoals2"
REGION="us-central1"
SERVICE_NAME="learning-goals-app"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"  # Replace this with your actual OpenAI API key when deploying

# Build and push the container image
echo "Building and pushing container image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Create a secret with the Firebase key
echo "Creating Firebase credentials secret..."
gcloud secrets create firebase-key --data-file=firebase-key.json --project=$PROJECT_ID || \
  (echo "Secret already exists, updating..." && \
  gcloud secrets versions add firebase-key --data-file=firebase-key.json --project=$PROJECT_ID)

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars "SECRET_KEY=learning-goals-secret-key-production" \
  --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY" \
  --set-env-vars "FIREBASE_STORAGE_BUCKET=learninggoals2.firebasestorage.app" \
  --set-env-vars "GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/firebase-key.json" \
  --service-account learning-goals-app@$PROJECT_ID.iam.gserviceaccount.com \
  --update-secrets="/tmp/keys/firebase-key.json=firebase-key:latest"

echo "Deployment completed successfully!"
echo "Your app is now available at: $(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')" 