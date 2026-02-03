# 🚀 Fire App Deployment Guide for PythonAnywhere

## Prerequisites
- PythonAnywhere account: https://www.pythonanywhere.com/user/rdhameja/
- All files from `/Users/rhuria/Downloads/Fire App/`

---

## Step 1: Upload Files to PythonAnywhere

### Method A: Using Web Interface (Recommended)

1. **Go to Files tab**: https://www.pythonanywhere.com/user/rdhameja/files/

2. **Navigate to your existing app directory** (likely `/home/rdhameja/` or wherever LineGuard is)

3. **Create/Update directory structure**:
```
/home/rdhameja/FireApp/
├── app.py
├── risk_model.py
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
└── models/ (if you have trained models)
```

4. **Upload files**:
   - Click "Upload a file" for each file
   - Create directories first, then upload into them
   - OR use "Open Bash console here" and upload via git/wget

### Method B: Using Bash Console

1. **Open Bash console**: https://www.pythonanywhere.com/user/rdhameja/consoles/

2. **Run these commands**:
```bash
# Navigate to home directory
cd ~

# Backup old LineGuard (optional)
mv lineguard lineguard_backup

# Create new directory
mkdir -p FireApp/templates FireApp/static/css FireApp/static/js FireApp/models

# Now upload files via Files tab or git clone
```

---

## Step 2: Install Dependencies

1. **Open Bash console**: https://www.pythonanywhere.com/user/rdhameja/consoles/

2. **Create virtual environment** (if not exists):
```bash
cd ~/FireApp
mkvirtualenv --python=/usr/bin/python3.10 fireapp-env
```

3. **Install packages**:
```bash
workon fireapp-env
pip install flask requests numpy pandas scikit-learn shap
```

**Package Versions** (tested):
- flask==3.0.0
- requests==2.31.0
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0
- shap==0.42.1

---

## Step 3: Configure Web App

### A. Update WSGI Configuration

1. **Go to Web tab**: https://www.pythonanywhere.com/user/rdhameja/webapps/

2. **Click on your existing web app** (lineguard.pythonanywhere.com)

3. **Click "WSGI configuration file"** link (e.g., `/var/www/rdhameja_pythonanywhere_com_wsgi.py`)

4. **Replace ALL content** with:

```python
import sys
import os

# Add your project directory to sys.path
project_home = '/home/rdhameja/FireApp'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Activate virtual environment
activate_this = '/home/rdhameja/.virtualenvs/fireapp-env/bin/activate_this.py'
if os.path.exists(activate_this):
    with open(activate_this) as file_:
        exec(file_.read(), dict(__file__=activate_this))

# Import Flask app
from app import app as application

# IMPORTANT: Remove Flask's development server port
# PythonAnywhere handles this automatically
```

5. **Click "Save"** (top right)

### B. Update Static Files Mapping

1. **Still in Web tab**, scroll to **"Static files"** section

2. **Update or add**:
   - URL: `/static/`
   - Directory: `/home/rdhameja/FireApp/static/`

3. **Click "✓"** to save

### C. Set Virtual Environment Path

1. **In Web tab**, find **"Virtualenv"** section

2. **Enter path**: `/home/rdhameja/.virtualenvs/fireapp-env`

3. **Click "✓"**

---

## Step 4: Modify app.py for Production

**IMPORTANT**: PythonAnywhere doesn't use `app.run()` with ports.

1. **Open** `/home/rdhameja/FireApp/app.py` in PythonAnywhere editor

2. **Find the last lines**:
```python
if __name__ == "__main__":
    app.run(debug=True, port=5001)
```

3. **Change to**:
```python
if __name__ == "__main__":
    app.run(debug=False)  # Remove port, set debug=False for production
```

**OR** just remove the entire `if __name__ == "__main__":` block since WSGI handles it.

---

## Step 5: Reload and Test

1. **Go to Web tab**: https://www.pythonanywhere.com/user/rdhameja/webapps/

2. **Click the big green "Reload" button** at the top

3. **Wait 5-10 seconds**

4. **Visit**: https://rdhameja.pythonanywhere.com

5. **Should see**: Fire App with map and panels! 🎉

---

## Step 6: Troubleshooting

### Check Error Logs

If the site doesn't load:

1. **In Web tab**, click **"Error log"** link
2. **Look for errors** (Python exceptions, import errors, etc.)
3. **Common issues**:

#### Issue 1: ImportError (missing packages)
```bash
workon fireapp-env
pip install <missing-package>
# Then reload web app
```

#### Issue 2: File not found
- Check file paths in WSGI config
- Ensure all files uploaded correctly
- Run: `ls -la ~/FireApp` in Bash

#### Issue 3: "No module named 'app'"
- Check WSGI config path is correct: `/home/rdhameja/FireApp`
- Verify `app.py` exists in that directory

#### Issue 4: Static files not loading (CSS/JS)
- Check Static files mapping in Web tab
- Path should be: `/home/rdhameja/FireApp/static/`
- Reload web app

### Check Server Log

1. **In Web tab**, click **"Server log"** link
2. **Look for startup messages**

### Test in Bash Console

```bash
cd ~/FireApp
workon fireapp-env
python3 app.py
# Should start without errors (Ctrl+C to stop)
```

---

## Step 7: Verify Everything Works

✅ **Checklist**:
- [ ] Map loads with California visible
- [ ] Left panel (tabs) shows City Focus & Predictor
- [ ] Right panel shows Live Metrics
- [ ] Clicking "Scan Area" works (selects city)
- [ ] Time Predictor slider works
- [ ] Markers appear on map
- [ ] No console errors (F12 → Console tab in browser)

---

## Step 8: Add NASA FIRMS API Key (When Ready)

Once you get your NASA FIRMS API key:

1. **Edit** `/home/rdhameja/FireApp/app.py`

2. **Add after line 29** (after USGS_ELEVATION_URL):
```python
# === NASA FIRMS API Configuration ===
NASA_FIRMS_KEY = "YOUR_NASA_FIRMS_KEY_HERE"
NASA_FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
```

3. **Reload web app**

---

## Quick Reference Commands

```bash
# SSH/Bash Console Commands
cd ~/FireApp                     # Navigate to app
workon fireapp-env              # Activate virtualenv
pip list                        # List installed packages
pip install <package>           # Install new package
python3 app.py                  # Test app locally
ls -la                          # List files
tail -f /var/log/rdhameja.pythonanywhere.com.error.log  # Watch error log
```

---

## Important URLs

- **Your App**: https://rdhameja.pythonanywhere.com
- **Web Config**: https://www.pythonanywhere.com/user/rdhameja/webapps/
- **Files**: https://www.pythonanywhere.com/user/rdhameja/files/
- **Consoles**: https://www.pythonanywhere.com/user/rdhameja/consoles/

---

## Support

If you encounter issues:
1. Check error logs first
2. Verify file paths
3. Confirm all dependencies installed
4. Try reloading web app
5. Restart console if needed

---

*Last updated: December 30, 2025*
*App: FireGuardAI - Power Line Vegetation Monitoring*

