#!/bin/bash
set -e

# Load sensitive configuration from deploy-config.sh (if it exists)
if [ -f "./deploy-config.sh" ]; then
  echo "Loading configuration from deploy-config.sh..."
  source ./deploy-config.sh
else
  echo "Warning: deploy-config.sh not found. You must set DEPLOY_OPENAI_API_KEY manually."
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
IMAGE_TAG="gcr.io/$PROJECT_ID/$SERVICE_NAME:latest"

# Verify API key is set
if [ -z "${OPENAI_API_KEY}" ]; then
  echo "Error: OpenAI API key is not set."
  exit 1
fi

# Configure Docker to use gcloud credentials
echo "⚙️  Configuring Docker authentication..."
gcloud auth configure-docker --quiet

# Check if base image exists locally, if not pull it
echo "📦 Checking local base image cache..."
if ! docker image inspect gcr.io/$PROJECT_ID/learning-goals-base:latest &>/dev/null; then
  echo "Pulling base image to local cache (one-time setup)..."
  docker pull gcr.io/$PROJECT_ID/learning-goals-base:latest
else
  echo "✅ Base image found in local cache!"
fi

# Build app image locally (super fast with cached base)
echo "🚀 Building app image locally (using cached base)..."
docker build -f Dockerfile.app -t $IMAGE_TAG .

# Push only the app image (fast - only new layers)
echo "📤 Pushing app image..."
docker push $IMAGE_TAG

# Create Firebase secret if needed
echo "🔐 Checking Firebase credentials..."
if ! gcloud secrets describe firebase-key --project=$PROJECT_ID &>/dev/null; then
  echo "Creating Firebase credentials secret..."
  gcloud secrets create firebase-key --data-file=firebase-key.json --project=$PROJECT_ID
else
  echo "✅ Firebase secret exists"
fi

# Deploy to Cloud Run
echo "☁️  Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image $IMAGE_TAG \
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
  --service-account learning-goals-app@$PROJECT_ID.iam.gserviceaccount.com \
  --update-secrets="/tmp/keys/firebase-key.json=firebase-key:latest"

echo "🎉 Ultra-fast deployment completed!"
echo "🌐 Your app: $(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --format 'value(status.url)')" 