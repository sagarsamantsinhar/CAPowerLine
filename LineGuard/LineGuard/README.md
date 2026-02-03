# 🔥 Power Line Vegetation Monitor

A real-time web application for monitoring vegetation growth near California transmission lines to predict and prevent fire hazards.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🗺️ **Interactive Map Visualization** - Real-time mapping of California transmission lines using Leaflet.js
- 📊 **Live Statistics Dashboard** - Track active alerts, monitored zones, and average vegetation height
- 🌳 **Vegetation Growth Simulation** - 31-day predictive modeling of vegetation growth
- ⚠️ **Smart Alert System** - Automatic notifications when vegetation clearance falls below safe thresholds
- 🎯 **Location Search** - Search by city name or custom coordinates
- ⏯️ **Time Controls** - Play, pause, and navigate through time-based simulations
- 🎨 **Modern UI/UX** - Glassmorphism design with smooth animations and responsive layout
- 📱 **Mobile Responsive** - Optimized for desktop, tablet, and mobile devices

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or download the repository**

2. **Navigate to the project directory**
   ```bash
   cd "/Users/rhuria/Downloads/Fire App"
   ```

3. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   ```

4. **Activate the virtual environment**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser and navigate to**
   ```
   http://localhost:5000
   ```

3. **Grant notification permissions** (optional)
   - Allow browser notifications to receive real-time alerts

## 📖 How to Use

### Navigation

- **🎯 Location Search**: Use the left sidebar to search by city or enter custom coordinates
- **⏯️ Time Controls**: Use play, pause, and reset buttons to control the simulation
- **📅 Date Slider**: Drag the slider to navigate through different dates
- **👁️ Toggle Panels**: Click the menu button (top right) to show/hide side panels

### Understanding the Map

- **🟢 Green Zones**: Low vegetation risk
- **🟡 Yellow Zones**: Moderate vegetation risk  
- **🔴 Red Zones**: High vegetation risk / Active alerts
- **⚡ Black Lines**: Transmission lines (hover for details)
- **⚠️ Alert Markers**: Locations requiring immediate attention

### Statistics Panel

- **Active Alerts**: Number of current high-risk zones
- **Monitored Zones**: Total vegetation zones being tracked
- **Avg Vegetation**: Average vegetation height across all zones

## 🛠️ Technology Stack

### Backend
- **Flask** - Python web framework
- **Requests** - HTTP library for API calls
- **ArcGIS API** - Real transmission line data

### Frontend
- **Leaflet.js** - Interactive mapping library
- **Bootstrap 5** - Responsive UI framework
- **Font Awesome 6** - Icon library
- **Google Fonts** - Poppins & Inter typefaces

### Features
- Glassmorphism design with CSS backdrop filters
- CSS3 animations and transitions
- Real-time data visualization
- Browser notification API

## 📊 Data Sources

The application fetches real-time transmission line data from:
- **ArcGIS Feature Server** - California power line geometries
- Includes voltage (kV), line length, and operational status

## 🎨 Design Features

- **Glassmorphism Effects** - Modern frosted glass appearance
- **Gradient Animations** - Dynamic background gradients
- **Smooth Transitions** - Cubic bezier animations for all interactions
- **Custom Icons** - Font Awesome integration throughout
- **Responsive Layout** - Mobile-first design approach
- **Dark Mode Ready** - Easy to implement dark theme

## ⚙️ Configuration

### Thresholds (in `app.py`)

```python
LINE_HEIGHT_M = 8.0           # Height of transmission lines (meters)
THRESHOLD_DISTANCE = 6.0       # Alert threshold (meters)
DAYS = 31                      # Simulation duration (days)
```

### Simulation Parameters

The app simulates vegetation growth using:
- Random initial heights (0.2m - 0.6m)
- Variable growth rates (0.05m - 0.35m per day)
- Small random fluctuations for realism

## 🔧 Customization

### Adding New Cities

Edit `static/js/main.js` and add to the `cityCenters` object:

```javascript
cityCenters = {
    newCity: [latitude, longitude],
    // ... existing cities
}
```

Then add the option to `templates/index.html`:

```html
<option value="newCity">New City Name</option>
```

### Changing Color Schemes

Edit CSS variables in `static/css/style.css`:

```css
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --danger-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    /* ... other variables */
}
```

## 📝 API Endpoints

- `GET /` - Main application page
- `GET /api/metadata` - Returns lines, zones, and date list
- `GET /api/state?date=YYYY-MM-DD` - Returns zone and alert data for specific date

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9
```

### Module Not Found Error
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Map Not Loading
- Check internet connection (map tiles require network access)
- Check browser console for errors
- Verify ArcGIS API endpoint is accessible

## 📄 License

This project is available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Support

For support or questions, please open an issue in the repository.

---

**Made with ❤️ for wildfire prevention and public safety**






