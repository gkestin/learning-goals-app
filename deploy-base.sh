#!/bin/bash
set -e

echo "Building base image with heavy ML dependencies..."
echo "This will take 10-20 minutes but only needs to be done when requirements.txt changes."

# Configuration
PROJECT_ID="learninggoals2"

# Build the base image with all the heavy dependencies
echo "Starting base image build..."
gcloud builds submit --config=cloudbuild.base.yaml

echo "Base image build completed successfully!"
echo "Now you can use ./deploy-fast.sh for quick deployments!" 