# Streamlit Cloud Deployment Guide

## 🚀 Step-by-Step Deployment Process

### Prerequisites
- ✅ GitHub account
- ✅ Streamlit Cloud account (free at share.streamlit.io)
- ✅ Trained model files in `artifacts/` folder

---

## 📋 Pre-Deployment Checklist

### 1. Verify Required Files
```bash
# Check if all files exist
ls app.py
ls requirements.txt
ls packages.txt
ls .streamlit/config.toml
ls artifacts/best_model.pkl
ls artifacts/preprocessor.pkl
```

### 2. Test Locally First
```bash
# Run dashboard locally
streamlit run app.py

# Verify:
# - Dashboard loads without errors
# - All sections render correctly
# - Predictions work
# - Charts display properly
```

### 3. Update .gitignore (IMPORTANT!)
**Remove artifacts from .gitignore to deploy models:**

```bash
# Edit .gitignore and comment out or remove:
# artifacts/

# OR create artifacts/.gitignore with:
*
!best_model.pkl
!preprocessor.pkl
!.gitignore
```

This ensures model files are committed to GitHub.

---

## 🌐 Deployment Steps

### Option A: Streamlit Cloud (Recommended)

#### Step 1: Prepare Repository
```bash
# Add all files
git add app.py requirements.txt packages.txt .streamlit/

# Add model files (CRITICAL!)
git add artifacts/best_model.pkl artifacts/preprocessor.pkl

# Commit
git commit -m "Deploy Streamlit dashboard with trained models"

# Push to GitHub
git push origin main
```

#### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `rkpcode/Retail_Sales_Intelligence_System`
4. Set main file path: `app.py`
5. Click "Deploy"

#### Step 3: Monitor Deployment
- Watch build logs for errors
- Common issues:
  - Missing dependencies → Check `requirements.txt`
  - Model file not found → Ensure artifacts committed
  - Import errors → Verify package versions

#### Step 4: Test Live App
- Click the generated URL (e.g., `https://your-app.streamlit.app`)
- Test all features:
  - KPI metrics display
  - What-If Simulator works
  - Charts render
  - Predictions return results

---

### Option B: Hugging Face Spaces

#### Step 1: Create Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Name: `retail-intelligence-system`
4. SDK: Select "Streamlit"
5. Click "Create Space"

#### Step 2: Upload Files
```bash
# Clone the Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/retail-intelligence-system
cd retail-intelligence-system

# Copy files
cp /path/to/app.py .
cp /path/to/requirements.txt .
cp -r /path/to/.streamlit .
cp -r /path/to/artifacts .

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

#### Step 3: Configure Space
- Space will auto-build and deploy
- Access at: `https://huggingface.co/spaces/YOUR_USERNAME/retail-intelligence-system`

---

### Option C: Docker Deployment

#### Step 1: Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
COPY packages.txt .
RUN apt-get update && xargs -a packages.txt apt-get install -y

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY .streamlit .streamlit/
COPY artifacts artifacts/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Step 2: Build and Run
```bash
# Build image
docker build -t retail-intelligence .

# Run container
docker run -p 8501:8501 retail-intelligence

# Access at http://localhost:8501
```

#### Step 3: Deploy to Cloud
```bash
# Push to Docker Hub
docker tag retail-intelligence YOUR_USERNAME/retail-intelligence
docker push YOUR_USERNAME/retail-intelligence

# Deploy to cloud provider (AWS, GCP, Azure, etc.)
```

---

## 🔧 Troubleshooting

### Issue 1: Model File Not Found
**Error:** `FileNotFoundError: artifacts/best_model.pkl`

**Solution:**
```bash
# Ensure artifacts are committed
git add -f artifacts/best_model.pkl artifacts/preprocessor.pkl
git commit -m "Add model files"
git push
```

### Issue 2: Import Errors
**Error:** `ModuleNotFoundError: No module named 'plotly'`

**Solution:**
```bash
# Update requirements.txt
echo "plotly>=5.17.0" >> requirements.txt
git add requirements.txt
git commit -m "Add missing dependency"
git push
```

### Issue 3: Memory Limit Exceeded
**Error:** `MemoryError` or app crashes

**Solution:**
- Optimize model size (use model compression)
- Reduce chart data points
- Use Streamlit Cloud's paid tier for more resources

### Issue 4: Slow Loading
**Solution:**
```python
# Add caching to app.py
@st.cache_resource
def load_models():
    # ... existing code
```

---

## 📊 Post-Deployment

### 1. Update README
```markdown
## 🌐 Live Demo
**Dashboard:** https://your-app.streamlit.app
```

### 2. Share on LinkedIn
```
🚀 Excited to share my Retail Sales Intelligence System!

✅ Real-time profitability predictions
✅ Interactive What-If simulator
✅ Product category analytics
✅ 85%+ model accuracy

Try it live: [Your URL]

#DataScience #MachineLearning #Streamlit #Python
```

### 3. Monitor Usage
- Check Streamlit Cloud analytics
- Monitor error logs
- Gather user feedback

---

## 🎯 Success Checklist

- [ ] Dashboard loads in <5 seconds
- [ ] All 4 sections render correctly
- [ ] Predictions work without errors
- [ ] Charts are interactive
- [ ] Mobile-responsive design
- [ ] No console errors
- [ ] Model files load successfully
- [ ] URL is shareable and public

---

## 📝 Important Notes

1. **Model Files:** MUST be committed to GitHub for Streamlit Cloud
2. **File Size:** Keep artifacts <100MB (GitHub limit)
3. **Dependencies:** Pin versions in requirements.txt
4. **Secrets:** Use Streamlit secrets for API keys (if needed)
5. **Updates:** Push to GitHub → Auto-redeploys on Streamlit Cloud

---

## 🔗 Useful Links

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)
- [Docker Documentation](https://docs.docker.com/)

---

**Ready to Deploy?** Follow Option A (Streamlit Cloud) for fastest deployment! 🚀
