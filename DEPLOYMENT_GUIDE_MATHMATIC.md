# 🚀 MathMatic.org Deployment Guide

This guide walks you through deploying the Learning Goals application to your custom domain **MathMatic.org** using Google Cloud Run and Namecheap DNS.

## 📋 Prerequisites

- [x] Google Cloud account with billing enabled
- [x] Domain **MathMatic.org** registered with Namecheap
- [x] Google Cloud CLI (`gcloud`) installed and authenticated
- [x] Docker installed (for local builds)
- [x] OpenAI API key

## 🔧 Step 1: Namecheap DNS Configuration

1. **Log into Namecheap**
   - Go to [Namecheap Domain List](https://ap.www.namecheap.com/domains/list/)
   - Click **"Manage"** next to `MathMatic.org`

2. **Go to Advanced DNS**
   - Click on the **"Advanced DNS"** tab

3. **Add DNS Records**
   
   **Delete any existing A and CNAME records first**, then add these:

   ```
   Type: CNAME
   Host: www
   Value: ghs.googlehosted.com.
   TTL: Automatic
   ```

   ```
   Type: A
   Host: @
   Value: 216.239.32.21
   TTL: Automatic
   ```

   ```
   Type: A
   Host: @
   Value: 216.239.34.21
   TTL: Automatic
   ```

   ```
   Type: A
   Host: @
   Value: 216.239.36.21
   TTL: Automatic
   ```

   ```
   Type: A
   Host: @
   Value: 216.239.38.21
   TTL: Automatic
   ```

4. **Save Changes**
   - DNS propagation can take 24-48 hours
   - Check status at [DNS Checker](https://dnschecker.org)

## 🚀 Step 2: Deploy to Google Cloud

### Option A: Full Deployment (First Time)

If this is your first deployment or you've changed `requirements.txt`:

```bash
# 1. Build base image (needed for heavy ML dependencies)
./deploy-base.sh

# 2. Deploy with custom domain setup
./deploy.sh
```

### Option B: Quick Updates (Recommended)

For code changes without dependency updates:

```bash
./deploy-ultra-fast.sh
```

### Option C: Domain Setup Only

If you've already deployed but need to set up the domain:

```bash
./setup-domain.sh
```

## 🌐 Step 3: Verify Deployment

After deployment completes, your app will be available at:

- **https://mathmatic.org** (primary domain)
- **https://www.mathmatic.org** (www subdomain)
- Direct Cloud Run URL (as backup)

### Testing Checklist:

- [ ] Visit https://mathmatic.org - should load the app
- [ ] Visit https://www.mathmatic.org - should load the app
- [ ] HTTP requests should automatically redirect to HTTPS
- [ ] All features working (PDF upload, clustering, etc.)

## 🔒 Step 4: SSL Certificate

Google Cloud Run automatically provisions SSL certificates for your custom domain. This usually takes 10-15 minutes after DNS propagation.

## 🛠️ Troubleshooting

### DNS Issues

**Problem**: Domain not resolving
**Solution**: 
- Wait 24-48 hours for full DNS propagation
- Check DNS records are correct in Namecheap
- Use `nslookup mathmatic.org` to check DNS resolution

### SSL Certificate Issues

**Problem**: "Not secure" warning in browser
**Solution**:
- Wait 10-15 minutes for certificate provisioning
- Check domain mapping status:
  ```bash
  gcloud run domain-mappings list --region us-central1
  ```

### Application Errors

**Problem**: 500 Internal Server Error
**Solution**:
- Check Cloud Run logs:
  ```bash
  gcloud logs read --project learninggoals2 --service learning-goals-app
  ```
- Verify environment variables are set correctly

## 📁 Project Structure

Key files for deployment:

```
├── deploy.sh                    # Full deployment with domain setup
├── deploy-ultra-fast.sh         # Quick updates
├── setup-domain.sh             # Domain configuration only
├── deploy-base.sh              # Base image (for dependency changes)
├── config.py                   # App configuration with domain settings
├── app/__init__.py             # Flask app with HTTPS redirect
├── requirements.txt            # Dependencies (includes flask-cors)
└── Dockerfile                  # Container configuration
```

## 🔄 Regular Updates

For ongoing development:

1. **Code changes**: Use `./deploy-ultra-fast.sh` (2-3 minutes)
2. **Dependency changes**: Use `./deploy-base.sh` then `./deploy-ultra-fast.sh`
3. **Configuration changes**: Use `./deploy.sh`

## 🎯 Production URLs

- **Primary**: https://mathmatic.org
- **Secondary**: https://www.mathmatic.org
- **Admin**: Google Cloud Console for monitoring and logs

## 🆘 Support

If you encounter issues:

1. Check the logs: `gcloud logs read --project learninggoals2`
2. Verify DNS: https://dnschecker.org
3. Test direct Cloud Run URL first
4. Ensure all environment variables are set

---

**🎉 Congratulations!** Your Learning Goals application is now live at **https://mathmatic.org** 