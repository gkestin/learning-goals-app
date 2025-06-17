#!/bin/bash
set -e

# Configuration
PROJECT_ID="learninggoals2"
REGION="us-central1"
SERVICE_NAME="learning-goals-app"
CUSTOM_DOMAIN="mathmatic.org"

echo "============================================"
echo "Setting up custom domain for Learning Goals"
echo "============================================"
echo "Domain: $CUSTOM_DOMAIN"
echo "Service: $SERVICE_NAME"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Check if service exists
echo "Checking if Cloud Run service exists..."
if ! gcloud run services describe $SERVICE_NAME --region $REGION --quiet > /dev/null 2>&1; then
    echo "Error: Cloud Run service '$SERVICE_NAME' not found in region '$REGION'"
    echo "Please deploy your service first using: ./deploy.sh"
    exit 1
fi

echo "✅ Service found"

# Set up custom domain mapping
echo ""
echo "Setting up domain mappings..."

# Map the apex domain (mathmatic.org)
echo "🔗 Mapping $CUSTOM_DOMAIN..."
if gcloud run domain-mappings create --service $SERVICE_NAME --domain $CUSTOM_DOMAIN --region $REGION 2>/dev/null; then
    echo "✅ Successfully mapped $CUSTOM_DOMAIN"
else
    echo "⚠️  Domain mapping may already exist for $CUSTOM_DOMAIN"
fi

# Map the www subdomain
echo "🔗 Mapping www.$CUSTOM_DOMAIN..."
if gcloud run domain-mappings create --service $SERVICE_NAME --domain www.$CUSTOM_DOMAIN --region $REGION 2>/dev/null; then
    echo "✅ Successfully mapped www.$CUSTOM_DOMAIN"
else
    echo "⚠️  Domain mapping may already exist for www.$CUSTOM_DOMAIN"
fi

# Get domain mapping status
echo ""
echo "📋 Current domain mappings:"
gcloud run domain-mappings list --region $REGION --filter="metadata.name:$CUSTOM_DOMAIN OR metadata.name:www.$CUSTOM_DOMAIN" --format="table(metadata.name,status.url,status.conditions[0].type:label=STATUS)"

echo ""
echo "============================================"
echo "🎉 Domain setup completed!"
echo "============================================"
echo ""
echo "Your Learning Goals app will be available at:"
echo "• https://$CUSTOM_DOMAIN"
echo "• https://www.$CUSTOM_DOMAIN"
echo ""
echo "📝 DNS Configuration for Namecheap:"
echo "============================================"
echo "In your Namecheap Advanced DNS settings, add these records:"
echo ""
echo "Type: CNAME"
echo "Host: www"
echo "Value: ghs.googlehosted.com."
echo "TTL: Automatic"
echo ""
echo "Type: A"
echo "Host: @"
echo "Value: 216.239.32.21"
echo "TTL: Automatic"
echo ""
echo "Type: A"
echo "Host: @"
echo "Value: 216.239.34.21"
echo "TTL: Automatic"
echo ""
echo "Type: A"
echo "Host: @"
echo "Value: 216.239.36.21"
echo "TTL: Automatic"
echo ""
echo "Type: A"
echo "Host: @"
echo "Value: 216.239.38.21"
echo "TTL: Automatic"
echo ""
echo "⏱️  DNS propagation can take up to 24-48 hours"
echo "🔍 You can check propagation status at: https://dnschecker.org"
echo ""
echo "============================================" 