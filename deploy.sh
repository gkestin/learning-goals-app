#!/bin/bash
set -e

# Load sensitive configuration from deploy-config.sh (if it exists)
if [ -f "./deploy-config.sh" ]; then
  echo "Loading configuration from deploy-config.sh..."
  source ./deploy-config.sh
else
  echo "Warning: deploy-config.sh not found. You must set DEPLOY_OPENAI_API_KEY manually."
  # Prompt for API key if not set
  if [ -z "${DEPLOY_OPENAI_API_KEY}" ]; then
    read -sp "Enter your OpenAI API key: " DEPLOY_OPENAI_API_KEY
    echo
  fi
fi

# Configuration
PROJECT_ID="learninggoals2"
REGION="us-central1"
SERVICE_NAME="learning-goals-app"
OPENAI_API_KEY="${DEPLOY_OPENAI_API_KEY}"
CUSTOM_DOMAIN="mathmatic.org"

# Verify API key is set
if [ -z "${OPENAI_API_KEY}" ]; then
  echo "Error: OpenAI API key is not set. Please update deploy-config.sh or set DEPLOY_OPENAI_API_KEY environment variable."
  exit 1
fi

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
  --memory=4Gi \
  --cpu=2 \
  --timeout=3600 \
  --no-use-http2 \
  --set-env-vars "SECRET_KEY=learning-goals-secret-key-production" \
  --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY" \
  --set-env-vars "FIREBASE_STORAGE_BUCKET=learninggoals2.firebasestorage.app" \
  --set-env-vars "GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/firebase-key.json" \
  --set-env-vars "CUSTOM_DOMAIN=$CUSTOM_DOMAIN" \
  --set-env-vars "FORCE_HTTPS=true" \
  --service-account learning-goals-app@$PROJECT_ID.iam.gserviceaccount.com \
  --update-secrets="/tmp/keys/firebase-key.json=firebase-key:latest"

# Get the Cloud Run service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')
echo "Service deployed at: $SERVICE_URL"

# Set up custom domain mapping
echo "Setting up custom domain mapping..."

# Map the apex domain (mathmatic.org)
echo "Mapping $CUSTOM_DOMAIN..."
gcloud run domain-mappings create --service $SERVICE_NAME --domain $CUSTOM_DOMAIN --region $REGION || \
  echo "Domain mapping may already exist for $CUSTOM_DOMAIN"

# Map the www subdomain
echo "Mapping www.$CUSTOM_DOMAIN..."
gcloud run domain-mappings create --service $SERVICE_NAME --domain www.$CUSTOM_DOMAIN --region $REGION || \
  echo "Domain mapping may already exist for www.$CUSTOM_DOMAIN"

echo ""
echo "============================================"
echo "Deployment completed successfully!"
echo "============================================"
echo "Your app is now available at:"
echo "• https://$CUSTOM_DOMAIN"
echo "• https://www.$CUSTOM_DOMAIN"
echo "• $SERVICE_URL (Cloud Run direct URL)"
echo ""
echo "Make sure your Namecheap DNS is configured as follows:"
echo "Type: CNAME, Host: www, Value: ghs.googlehosted.com."
echo "Type: A, Host: @, Value: 216.239.32.21"
echo "Type: A, Host: @, Value: 216.239.34.21"
echo "Type: A, Host: @, Value: 216.239.36.21"
echo "Type: A, Host: @, Value: 216.239.38.21"
echo "============================================" 