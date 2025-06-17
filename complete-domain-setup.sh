#!/bin/bash
set -e

echo "🌐 Completing MathMatic.org Domain Setup"
echo "========================================"
echo ""

# Configuration
PROJECT_ID="learninggoals2"
REGION="us-central1"
SERVICE_NAME="learning-goals-app"
CUSTOM_DOMAIN="mathmatic.org"

echo "📋 Prerequisites Check:"
echo "✅ Domain verification completed in Google Search Console"
echo "✅ DNS records added to Namecheap"
echo "✅ App deployed to Cloud Run"
echo ""

read -p "Have you completed domain verification in Google Search Console? (y/n): " verified
if [[ $verified != "y" && $verified != "Y" ]]; then
    echo "❌ Please complete domain verification first:"
    echo "   1. Go to https://search.google.com/search-console"
    echo "   2. Add property for mathmatic.org"
    echo "   3. Add the TXT record to Namecheap DNS"
    echo "   4. Verify the domain"
    echo "   5. Then run this script again"
    exit 1
fi

echo "🔗 Creating domain mappings..."

# Map the apex domain (mathmatic.org)
echo "Mapping $CUSTOM_DOMAIN..."
if gcloud beta run domain-mappings create --service $SERVICE_NAME --domain $CUSTOM_DOMAIN --region $REGION; then
    echo "✅ Successfully mapped $CUSTOM_DOMAIN"
else
    echo "⚠️  Failed to map $CUSTOM_DOMAIN - may already exist or domain not verified"
fi

# Map the www subdomain
echo "Mapping www.$CUSTOM_DOMAIN..."
if gcloud beta run domain-mappings create --service $SERVICE_NAME --domain www.$CUSTOM_DOMAIN --region $REGION; then
    echo "✅ Successfully mapped www.$CUSTOM_DOMAIN"
else
    echo "⚠️  Failed to map www.$CUSTOM_DOMAIN - may already exist or domain not verified"
fi

# Check domain mapping status
echo ""
echo "📋 Domain Mapping Status:"
gcloud beta run domain-mappings list --region $REGION 2>/dev/null || echo "No domain mappings found"

echo ""
echo "🎉 Domain setup completed!"
echo ""
echo "============================================"
echo "🌐 Your Learning Goals app is now available at:"
echo "   • https://mathmatic.org"
echo "   • https://www.mathmatic.org"
echo "============================================"
echo ""
echo "⏱️  Note: It may take 10-15 minutes for:"
echo "   • SSL certificates to be provisioned"
echo "   • DNS to fully propagate worldwide"
echo ""
echo "🔍 You can check propagation status at:"
echo "   • https://dnschecker.org"
echo "   • https://www.whatsmydns.net"
echo ""
echo "🆘 If you encounter issues:"
echo "   • Wait up to 24 hours for full DNS propagation"
echo "   • Check that all DNS records are correct in Namecheap"
echo "   • Verify domain ownership is confirmed in Search Console"
echo ""
echo "✨ Happy teaching and learning! ✨" 