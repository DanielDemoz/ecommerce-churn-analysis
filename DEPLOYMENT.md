# 🚀 Deployment Guide - E-commerce Churn Analysis Dashboard

## 📋 Prerequisites

### System Requirements
- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **RAM:** Minimum 4GB (recommended: 8GB+)
- **Storage:** 1GB free space
- **Internet:** Required for package installation

### Required Packages
All dependencies are listed in `requirements.txt`:
```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
xgboost>=1.7.0
scipy>=1.9.0
statsmodels>=0.13.0
streamlit>=1.28.0
plotly>=5.15.0
openpyxl>=3.0.0
```

---

## 🖥️ Local Deployment

### Option 1: Automated Launch (Recommended)

#### Windows Users:
```bash
# Double-click the batch file
launch_dashboard.bat

# Or run from command prompt
python run_dashboard.py
```

#### Mac/Linux Users:
```bash
# Make executable and run
chmod +x launch_dashboard.sh
./launch_dashboard.sh

# Or run directly
python3 run_dashboard.py
```

### Option 2: Manual Setup

1. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

2. **Launch Dashboard:**
```bash
streamlit run dashboard.py
```

3. **Access Dashboard:**
- Open browser to: `http://localhost:8501`
- Dashboard will auto-open in default browser

---

## ☁️ Cloud Deployment

### Streamlit Cloud (Free)

1. **Prepare Repository:**
   - Upload code to GitHub
   - Ensure `requirements.txt` is in root directory
   - Add `config.toml` for custom settings

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub account
   - Select repository and branch
   - Click "Deploy"

3. **Configuration:**
   - App URL: `https://your-app-name.streamlit.app`
   - Auto-updates on code changes
   - Free tier: 1GB RAM, 1GB storage

### Heroku

1. **Create Heroku App:**
```bash
heroku create your-app-name
```

2. **Add Procfile:**
```
web: streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0
```

3. **Deploy:**
```bash
git add .
git commit -m "Deploy dashboard"
git push heroku main
```

### AWS EC2

1. **Launch EC2 Instance:**
   - Ubuntu 20.04 LTS
   - t3.medium or larger
   - Security group: Port 8501 open

2. **Setup Environment:**
```bash
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
```

3. **Run Dashboard:**
```bash
streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
```

---

## 🐳 Docker Deployment

### Create Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run:
```bash
# Build image
docker build -t churn-dashboard .

# Run container
docker run -p 8501:8501 churn-dashboard

# With volume for data persistence
docker run -p 8501:8501 -v $(pwd)/data:/app/data churn-dashboard
```

### Docker Compose:
```yaml
version: '3.8'
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

---

## 🔧 Configuration

### Environment Variables:
```bash
# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Application Settings
DASHBOARD_TITLE="E-commerce Churn Analysis"
DASHBOARD_THEME="light"
```

### Custom Configuration (`config.toml`):
```toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
enableCORS = false

[theme]
base = "light"
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

---

## 📊 Data Management

### Dataset Requirements:
- **Format:** Excel (.xlsx) or CSV (.csv)
- **Required Columns:** See dataset schema in README
- **Size:** Up to 200MB (configurable)

### Demo Data:
- Automatically created if no dataset found
- 1000 sample customers
- Realistic churn patterns
- Generated using `create_demo_data.py`

### Data Updates:
- Replace dataset file and restart dashboard
- No database required (file-based)
- Caching enabled for performance

---

## 🔒 Security Considerations

### Production Deployment:
1. **Authentication:** Add user login (Streamlit-Authenticator)
2. **HTTPS:** Use reverse proxy (Nginx) with SSL
3. **Firewall:** Restrict access to authorized IPs
4. **Data Privacy:** Ensure GDPR compliance

### Example Nginx Config:
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## 📈 Performance Optimization

### Caching:
- Streamlit caching enabled (`@st.cache_data`)
- Model training cached
- Data preprocessing cached

### Scaling:
- **Horizontal:** Multiple instances behind load balancer
- **Vertical:** Increase RAM for larger datasets
- **Database:** Use PostgreSQL for large datasets

### Monitoring:
- Streamlit built-in metrics
- Custom logging for model performance
- Health checks for containerized deployments

---

## 🐛 Troubleshooting

### Common Issues:

1. **Port Already in Use:**
```bash
# Find process using port 8501
lsof -i :8501
# Kill process
kill -9 <PID>
```

2. **Package Installation Errors:**
```bash
# Update pip
pip install --upgrade pip
# Install with no cache
pip install --no-cache-dir -r requirements.txt
```

3. **Dataset Loading Issues:**
- Check file format (Excel/CSV)
- Verify column names match expected schema
- Ensure file is in project directory

4. **Memory Issues:**
- Reduce dataset size
- Increase system RAM
- Use data sampling for large datasets

### Logs and Debugging:
```bash
# Enable debug mode
streamlit run dashboard.py --logger.level=debug

# Check Streamlit logs
tail -f ~/.streamlit/logs/streamlit.log
```

---

## 📞 Support

### Getting Help:
1. **Documentation:** Check README.md
2. **Issues:** GitHub Issues page
3. **Community:** Streamlit Community Forum
4. **Email:** [your-support-email]

### Contributing:
1. Fork repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request

---

## 🎯 Best Practices

### Development:
- Use virtual environments
- Test with different datasets
- Validate model performance
- Document customizations

### Production:
- Regular backups
- Monitor performance
- Update dependencies
- Security audits

### Maintenance:
- Regular data updates
- Model retraining
- Performance monitoring
- User feedback collection

---

**🚀 Happy Deploying! Your E-commerce Churn Analysis Dashboard is ready to help reduce customer churn!**
