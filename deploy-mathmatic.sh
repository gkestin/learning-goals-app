#!/bin/bash
set -e

echo "🚀 MathMatic.org Deployment Wizard"
echo "=================================="
echo "This script will deploy your Learning Goals app to mathmatic.org"
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check if gcloud is installed and authenticated
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: Google Cloud CLI (gcloud) is not installed"
    echo "   Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null; then
    echo "❌ Error: Not authenticated with Google Cloud"
    echo "   Run: gcloud auth login"
    exit 1
fi

echo "✅ Google Cloud CLI authenticated"

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Error: Docker is not running"
    echo "   Please start Docker Desktop and try again"
    exit 1
fi

echo "✅ Docker is running"

# Check for API key
if [ -f "./deploy-config.sh" ]; then
    echo "✅ Found deploy-config.sh"
    source ./deploy-config.sh
    if [ -z "${DEPLOY_OPENAI_API_KEY}" ]; then
        echo "❌ Error: DEPLOY_OPENAI_API_KEY not set in deploy-config.sh"
        exit 1
    fi
else
    echo "❓ deploy-config.sh not found"
    if [ -z "${DEPLOY_OPENAI_API_KEY}" ]; then
        echo "Please enter your OpenAI API key (it will be stored securely):"
        read -sp "OpenAI API Key: " DEPLOY_OPENAI_API_KEY
        echo ""
        
        # Create deploy-config.sh
        echo "Creating deploy-config.sh..."
        cat > deploy-config.sh << EOF
#!/bin/bash
# Deployment configuration
export DEPLOY_OPENAI_API_KEY="$DEPLOY_OPENAI_API_KEY"
EOF
        chmod +x deploy-config.sh
        echo "✅ API key saved to deploy-config.sh"
    fi
fi

echo ""
echo "🎯 Deployment Plan:"
echo "   Domain: mathmatic.org"
echo "   Project: learninggoals2"
echo "   Region: us-central1"
echo ""

# Ask for deployment type
echo "📦 Choose deployment type:"
echo "1. 🆕 First-time deployment (includes base image build - ~15 minutes)"
echo "2. ⚡ Quick update (app changes only - ~3 minutes)"
echo "3. 🌐 Domain setup only (if app is deployed but domain not configured)"
echo ""
read -p "Enter choice (1, 2, or 3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting first-time deployment..."
        echo "This will take approximately 15 minutes"
        echo ""
        
        # Build base image first
        echo "📦 Step 1/3: Building base image with ML dependencies..."
        if [ -f "./deploy-base.sh" ]; then
            ./deploy-base.sh
        else
            echo "❌ deploy-base.sh not found"
            exit 1
        fi
        
        echo ""
        echo "🚀 Step 2/3: Deploying application with domain setup..."
        ./deploy.sh
        
        echo ""
        echo "✅ Step 3/3: Verifying domain mappings..."
        ./setup-domain.sh
        ;;
    2)
        echo ""
        echo "⚡ Starting quick deployment..."
        ./deploy-ultra-fast.sh
        ;;
    3)
        echo ""
        echo "🌐 Setting up domain only..."
        ./setup-domain.sh
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "=============="
echo ""
echo "1. 🌐 Configure DNS in Namecheap:"
echo "   - Go to https://ap.www.namecheap.com/domains/list/"
echo "   - Click 'Manage' next to MathMatic.org"
echo "   - Go to 'Advanced DNS' tab"
echo "   - Add the DNS records as shown in DEPLOYMENT_GUIDE_MATHMATIC.md"
echo ""
echo "2. ⏳ Wait for DNS propagation (up to 24-48 hours)"
echo "   - Check status: https://dnschecker.org"
echo ""
echo "3. 🔒 SSL certificate will be auto-provisioned (10-15 minutes after DNS)"
echo ""
echo "4. 🎯 Your app will be live at:"
echo "   • https://mathmatic.org"
echo "   • https://www.mathmatic.org"
echo ""
echo "📖 For detailed instructions, see: DEPLOYMENT_GUIDE_MATHMATIC.md"
echo ""
echo "🆘 Need help? Check the troubleshooting section in the guide"
echo ""
echo "Happy deploying! 🚀" 