# Install required libraries with error handling


# Import libraries with fallbacks
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv
import keras
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from branca.element import Figure
import tempfile
import base64
from geopy.geocoders import Nominatim
import warnings
import io
import sys
import json
from PIL import Image
import threading
from gtts import gTTS
try:
    import pygame
except ImportError:
    pygame = None
    print("Pygame not installed. Voice alerts will be saved but not played.")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------- Configuration ---------------------
load_dotenv()
API_KEY = os.getenv("OWM_API_KEY", "ee346904823a7e8b151ec55fac07258c")

# Create directories safely
try:
    os.makedirs("/content/models", exist_ok=True)
    os.makedirs("/content/data", exist_ok=True)
    MODEL_PATH = "/content/models/cloudburst_model.weights.h5"
    RF_MODEL_PATH = "/content/models/rf_model.pkl"
    HISTORICAL_DATA_PATH = "/content/data/historical_cloudbursts.csv"
except Exception as e:
    print(f"Directory creation error: {e}")
    # Fallback to current directory
    MODEL_PATH = "cloudburst_model.weights.h5"
    RF_MODEL_PATH = "rf_model.pkl"
    HISTORICAL_DATA_PATH = "historical_cloudbursts.csv"

# --------------------- Voice Assistant System -------------
class VoiceAssistant:
    def __init__(self):
        self.audio_file = "alert.mp3"
        self.is_ready = False
        if pygame is not None:
            try:
                pygame.mixer.init()
                self.is_ready = True
                print("Audio initialized successfully")
            except Exception as e:
                print(f"Audio initialization error: {e}. Voice alerts will be saved but not played.")
        else:
            print("Pygame unavailable. Voice alerts will be saved but not played.")

    def generate_voice_alert(self, message):
        try:
            # Create voice alert
            tts = gTTS(text=message, lang='en', slow=False)

            # Save to in-memory bytes buffer
            audio_bytes = io.BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)

            # Save to file for potential replay
            with open(self.audio_file, "wb") as f:
                f.write(audio_bytes.getvalue())

            return audio_bytes
        except Exception as e:
            print(f"Voice alert generation error: {e}")
            return None

    def play_voice_alert(self):
        if not self.is_ready or not os.path.exists(self.audio_file):
            print("Cannot play alert: Audio system not ready or file missing")
            return False

        try:
            pygame.mixer.music.load(self.audio_file)
            pygame.mixer.music.play()

            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            return True
        except Exception as e:
            print(f"Voice alert playback error: {e}")
            return False

# --------------------- Data Simulation -------------------
def generate_realistic_data(num_samples=1000):
    np.random.seed(42)
    base_date = datetime.now() - timedelta(days=365)
    data = []

    # Generate historical cloudburst data
    historical_data = []
    for i in range(500):
        date = base_date - timedelta(days=np.random.randint(0, 3650))
        lat = np.random.uniform(8.0, 37.0)
        lon = np.random.uniform(68.0, 97.0)
        severity = np.random.choice(['Low', 'Moderate', 'High', 'Extreme'],
                                 p=[0.4, 0.3, 0.2, 0.1])
        casualties = 0
        if severity == 'Low':
            casualties = np.random.randint(0, 5)
        elif severity == 'Moderate':
            casualties = np.random.randint(3, 20)
        elif severity == 'High':
            casualties = np.random.randint(15, 50)
        else:
            casualties = np.random.randint(40, 200)

        economic_loss = casualties * np.random.uniform(0.5, 2.0) * 1000000

        historical_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'latitude': lat,
            'longitude': lon,
            'severity': severity,
            'casualties': casualties,
            'economic_loss': economic_loss,
            'location': f"Location {i+1}"
        })

    # Save historical data
    try:
        pd.DataFrame(historical_data).to_csv(HISTORICAL_DATA_PATH, index=False)
    except Exception as e:
        print(f"Error saving historical data: {e}")

    # Generate training data
    for _ in range(num_samples):
        sample = []
        for day in range(7):
            date = base_date + timedelta(days=day)
            day_of_year = date.timetuple().tm_yday
            temp = 10 + 15 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 3)
            humidity = 60 + 20 * np.cos(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5)
            pressure = 1013 + 5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 5)
            precipitation = np.random.gamma(shape=3.0, scale=4.0) if day_of_year % 90 < 30 else np.random.gamma(shape=2.0, scale=2.0)
            wind_speed = np.random.weibull(2.0) * 5 + 2 * np.sin(2 * np.pi * day_of_year / 365)
            cloud_cover = min(100, max(0, humidity - 20 + np.random.normal(0, 10)))
            uv_index = max(0, min(11, 11 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2)))
            visibility = max(0, min(20, 20 - (humidity/10) + np.random.normal(0, 2)))
            sample.append([
                max(0, precipitation),
                max(0, min(100, humidity)),
                max(800, pressure),
                temp,
                max(0, wind_speed),
                max(0, min(100, cloud_cover)),
                max(0, min(11, uv_index)),
                max(0.1, min(20, visibility))
            ])
        data.append(sample)
    X = np.array(data)
    y = (X[:, -1, 0] > 5).astype(int)
    return X, y.reshape(-1, 1)

# --------------------- Model Architecture ----------------
def create_advanced_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def create_rf_model():
    return RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# --------------------- Risk Alert System -----------------
class RiskAlertSystem:
    def __init__(self):
        self.alert_levels = {
            'Low': {'color': 'green', 'icon': '✓'},
            'Moderate': {'color': 'yellow', 'icon': '⚠️'},
            'High': {'color': 'orange', 'icon': '⚠️⚠️'},
            'Extreme': {'color': 'red', 'icon': '🔴🔴🔴'}
        }
        self.voice_assistant = VoiceAssistant()
        self.last_alert_message = ""

    def generate_alert_html(self, risk_level, probability, advice):
        alert_info = self.alert_levels.get(risk_level, self.alert_levels['Low'])

        # Generate voice alert for moderate risk
        if risk_level == "Moderate":
            self.last_alert_message = f"Moderate risk alert. Probability {probability*100:.1f}%. {advice}"
            audio_bytes = self.voice_assistant.generate_voice_alert(self.last_alert_message)

            # Play the alert automatically in a separate thread if audio is ready
            if audio_bytes and self.voice_assistant.is_ready:
                threading.Thread(target=self.voice_assistant.play_voice_alert).start()

        return f"""
        <div style="padding: 15px; background-color: {alert_info['color']}; border-radius: 10px; color: white; font-weight: bold; margin: 10px 0;">
            <h3>{alert_info['icon']} {risk_level.upper()} RISK ALERT - {probability*100:.1f}%</h3>
            <p>{advice}</p>
            <p>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """

    def get_evacuation_routes(self, lat, lon):
        return [
            {'name': 'Route A', 'distance': '2.3 km', 'estimated_time': '25 min', 'congestion': 'Low'},
            {'name': 'Route B', 'distance': '3.1 km', 'estimated_time': '30 min', 'congestion': 'Medium'},
            {'name': 'Route C', 'distance': '4.5 km', 'estimated_time': '40 min', 'congestion': 'High'}
        ]

    def get_emergency_contacts(self, city):
        contacts = {
            'Mumbai': {'disaster_management': '1916', 'police': '100', 'ambulance': '108', 'fire': '101'},
            'Delhi': {'disaster_management': '1077', 'police': '100', 'ambulance': '108', 'fire': '101'},
        }
        return contacts.get(city.title(), {'disaster_management': '1078', 'police': '100', 'ambulance': '108', 'fire': '101'})

# --------------------- Data Processing -------------------
def normalize_data(data):
    norms = np.array([[100, 100, 1100, 50, 50, 100, 11, 20]])
    return data / norms

def process_weather_data(raw_data):
    processed = []
    details = []
    for entry in raw_data['list'][:7]:
        dt = datetime.fromtimestamp(entry['dt'])

        weather = {
            'date': dt.strftime('%Y-%m-%d %H:%M'),
            'temp': entry['main']['temp'] - 273.15,
            'feels_like': entry['main']['feels_like'] - 273.15,
            'temp_min': entry['main']['temp_min'] - 273.15,
            'temp_max': entry['main']['temp_max'] - 273.15,
            'humidity': entry['main']['humidity'],
            'pressure': entry['main']['pressure'],
            'precipitation': entry['rain']['3h'] if 'rain' in entry else 0.0,
            'wind_speed': entry['wind']['speed'],
            'wind_deg': entry['wind'].get('deg', 0),
            'wind_gust': entry['wind'].get('gust', entry['wind']['speed'] * 1.5),
            'description': entry['weather'][0]['description'],
            'icon': entry['weather'][0]['icon']
        }

        weather['cloud_cover'] = entry.get('clouds', {}).get('all', 0)
        weather['uv_index'] = np.random.uniform(0, 11)
        weather['visibility'] = entry.get('visibility', 10000) / 1000

        processed.append([
            weather['precipitation'],
            weather['humidity'],
            weather['pressure'],
            weather['temp'],
            weather['wind_speed'],
            weather['cloud_cover'],
            weather['uv_index'],
            weather['visibility']
        ])
        details.append(weather)
    return normalize_data(np.array([processed])), details

def calculate_stats_and_trends(weather_data):
    stats = {}
    for key in ['temp', 'humidity', 'pressure', 'precipitation', 'wind_speed', 'cloud_cover', 'uv_index', 'visibility',
               'feels_like', 'temp_min', 'temp_max', 'wind_gust']:
        if all(key in w for w in weather_data):
            values = [w[key] for w in weather_data]
            stats[f'{key}_mean'] = np.mean(values)
            stats[f'{key}_median'] = np.median(values)
            stats[f'{key}_std'] = np.std(values)
            stats[f'{key}_max'] = np.max(values)
            stats[f'{key}_min'] = np.min(values)
            stats[f'{key}_trend'] = np.polyfit(range(len(values)), values, 1)[0]
            stats[f'{key}_range'] = np.max(values) - np.min(values)
            stats[f'{key}_variance'] = np.var(values)
            stats[f'{key}_25th'] = np.percentile(values, 25)
            stats[f'{key}_75th'] = np.percentile(values, 75)
            stats[f'{key}_iqr'] = stats[f'{key}_75th'] - stats[f'{key}_25th']

    if all('precipitation' in w for w in weather_data):
        precip = [w['precipitation'] for w in weather_data]
        stats['precipitation_days'] = sum(p > 0.1 for p in precip)
        stats['heavy_rain_days'] = sum(p > 10 for p in precip)
        stats['precipitation_intensity'] = np.sum(precip) / max(1, stats['precipitation_days'])

    if all('wind_speed' in w and 'wind_deg' in w for w in weather_data):
        wind_dirs = [w['wind_deg'] for w in weather_data]
        diffs = []
        for i in range(1, len(wind_dirs)):
            diff = abs(wind_dirs[i] - wind_dirs[i-1])
            if diff > 180:
                diff = 360 - diff
            diffs.append(diff)
        stats['wind_direction_change'] = np.mean(diffs) if diffs else 0

    if all('temp_max' in w and 'temp_min' in w for w in weather_data):
        stats['diurnal_variation'] = np.mean([w['temp_max'] - w['temp_min'] for w in weather_data])

    if all('temp' in w and 'humidity' in w for w in weather_data):
        instability = []
        for w in weather_data:
            instability.append((w['temp'] / 10) * (w['humidity'] / 50))
        stats['instability_index'] = np.mean(instability)

    return stats

def get_safety_advice(risk_level, probability, stats):
    base_advice = {
        'Low': f"Low risk ({probability*100:.1f}% chance). No immediate concern. Monitor weather updates.",
        'Moderate': f"Moderate risk ({probability*100:.1f}% chance). Prepare for possible heavy rain; avoid low-lying areas.",
        'High': f"High risk ({probability*100:.1f}% chance). Stay indoors if possible. Secure property and avoid travel.",
        'Extreme': f"EXTREME RISK ({probability*100:.1f}% chance). SEEK SHELTER IMMEDIATELY. Follow evacuation orders."
    }

    additional_advice = []

    if stats.get('precipitation_trend', 0) > 0.5:
        additional_advice.append("Rainfall intensity is increasing rapidly.")

    if stats.get('pressure_trend', 0) < -0.5:
        additional_advice.append("Falling pressure indicates worsening conditions.")

    if stats.get('wind_speed_max', 0) > 15:
        additional_advice.append("Strong winds expected. Secure loose objects.")

    if stats.get('instability_index', 0) > 5:
        additional_advice.append("Atmospheric conditions are unstable, increasing severe weather risk.")

    if risk_level in ['High', 'Extreme']:
        additional_advice.append("Charge phones and prepare emergency supplies.")
        additional_advice.append("Keep emergency contacts handy.")

    full_advice = base_advice.get(risk_level, "Check weather updates regularly.")
    if additional_advice:
        full_advice += " " + " ".join(additional_advice)

    return full_advice

# --------------------- Enhanced Visualizations ----------------------
def create_visualizations(weather_data, probability, city, stats, regional_data=None):
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=[w['date'] for w in weather_data],
                            y=[w['temp'] for w in weather_data],
                            name='Temperature (°C)', line=dict(color='red')))
    fig1.add_trace(go.Bar(x=[w['date'] for w in weather_data],
                         y=[w['precipitation'] for w in weather_data],
                         name='Precipitation (mm)', yaxis='y2'))
    fig1.update_layout(
        title='Temperature & Precipitation',
        yaxis2=dict(overlaying='y', side='right', title='Precipitation (mm)'),
        yaxis=dict(title='Temperature (°C)'),
        height=300
    )

    fig2 = px.line(pd.DataFrame(weather_data), x='date',
                  y=['humidity', 'pressure'],
                  title='Humidity & Pressure',
                  labels={'value': 'Value', 'variable': 'Parameter'},
                  height=300)

    fig3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Cloudburst Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen", 'name': 'No Rain'},
                {'range': [20, 40], 'color': "lightblue", 'name': 'Light Rain'},
                {'range': [40, 60], 'color': "blue", 'name': 'Moderate Rain'},
                {'range': [60, 80], 'color': "orange", 'name': 'Heavy Rain'},
                {'range': [80, 100], 'color': "red", 'name': 'Cloudburst'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig3.update_layout(height=300)

    fig4 = px.bar(pd.DataFrame(weather_data), x='date', y='wind_speed',
                 title='Wind Speed Analysis',
                 labels={'wind_speed': 'Wind Speed (m/s)'},
                 color='wind_speed',
                 height=300)

    fig5 = px.pie(names=[w['description'] for w in weather_data],
                 title='Weather Conditions Distribution',
                 height=300)

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(x=[w['date'] for w in weather_data],
                             y=[w['temp'] for w in weather_data],
                             name='Actual Temperature', line=dict(color='blue')))
    fig7.add_trace(go.Scatter(x=[w['date'] for w in weather_data],
                             y=[w['feels_like'] for w in weather_data],
                             name='Feels Like', line=dict(color='red', dash='dot')))
    fig7.update_layout(title='Actual vs Feels Like Temperature', height=300)

    fig8 = go.Figure()
    wind_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    wind_speeds = []

    for direction in range(8):
        speeds = [w['wind_speed'] for w in weather_data if
                 (direction * 45 - 22.5 <= w['wind_deg'] % 360 <= direction * 45 + 22.5) or
                 (direction == 0 and w['wind_deg'] % 360 >= 337.5)]
        if not speeds:
            speeds = [0]
        wind_speeds.append(np.mean(speeds))

    fig8.add_trace(go.Barpolar(
        r=wind_speeds,
        theta=wind_directions,
        name='Wind Speed',
        marker_color=['rgba(0,0,255,0.7)' if s < 5 else
                     'rgba(0,255,0,0.7)' if s < 10 else
                     'rgba(255,255,0,0.7)' if s < 15 else
                     'rgba(255,0,0,0.7)' for s in wind_speeds]
    ))
    fig8.update_layout(
        title='Wind Rose Chart',
        polar=dict(
            radialaxis=dict(range=[0, max(wind_speeds) * 1.2]),
        ),
        height=350
    )

    heat_indices = []
    for w in weather_data:
        T = w['temp']
        RH = w['humidity']
        if T > 20 and RH > 40:
            heat_index = -8.78469475556 + 1.61139411 * T + 2.33854883889 * RH - \
                         0.14611605 * T * RH - 0.012308094 * T**2 - \
                         0.0164248277778 * RH**2 + 0.002211732 * T**2 * RH + \
                         0.00072546 * T * RH**2 - 0.000003582 * T**2 * RH**2
        else:
            heat_index = T
        heat_indices.append(heat_index)

    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=[w['date'] for w in weather_data],
                             y=heat_indices,
                             name='Heat Index',
                             line=dict(color='orangered'),
                             fill='tozeroy'))
    fig9.update_layout(title='Heat Index (Perceived Temperature)', height=300)

    fig10 = go.Figure()
    pressure_values = [w['pressure'] for w in weather_data]
    dates = [w['date'] for w in weather_data]
    fig10.add_trace(go.Scatter(x=dates, y=pressure_values, mode='lines+markers', name='Pressure'))
    z = np.polyfit(range(len(pressure_values)), pressure_values, 1)
    p = np.poly1d(z)
    fig10.add_trace(go.Scatter(x=dates, y=p(range(len(pressure_values))),
                              mode='lines', name='Trend',
                              line=dict(color='red', dash='dash')))
    fig10.add_trace(go.Scatter(x=dates, y=[1013.25] * len(dates),
                              mode='lines', name='Standard Pressure',
                              line=dict(color='green', dash='dot')))
    for i in range(1, len(pressure_values)):
        if pressure_values[i] - pressure_values[i-1] < -3:
            fig10.add_annotation(x=dates[i], y=pressure_values[i],
                               text="Rapid Drop",
                               showarrow=True, arrowhead=1)
    fig10.update_layout(title='Pressure Trend Analysis (hPa)', height=300)

    categories = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation',
                 'Cloud Cover', 'Pressure', 'UV Index', 'Visibility']
    avg_temp = np.mean([w['temp'] for w in weather_data]) / 40
    avg_humidity = np.mean([w['humidity'] for w in weather_data]) / 100
    avg_wind = np.mean([w['wind_speed'] for w in weather_data]) / 20
    avg_precip = np.mean([w['precipitation'] for w in weather_data]) / 20
    avg_cloud = np.mean([w['cloud_cover'] for w in weather_data]) / 100
    avg_pressure = (np.mean([w['pressure'] for w in weather_data]) - 970) / 70
    avg_uv = np.mean([w['uv_index'] for w in weather_data]) / 11
    avg_vis = np.mean([w['visibility'] for w in weather_data]) / 20

    values = [max(0, min(1, v)) for v in [avg_temp, avg_humidity, avg_wind, avg_precip, avg_cloud, avg_pressure, avg_uv, avg_vis]]

    fig11 = go.Figure()
    fig11.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Conditions'
    ))
    danger_values = [0.6, 0.8, 0.4, 0.7, 0.9, 0.3, 0.8, 0.4]
    fig11.add_trace(go.Scatterpolar(
        r=danger_values,
        theta=categories,
        fill='toself',
        name='Risk Threshold',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='red')
    ))
    fig11.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title="Weather Parameters Radar Analysis",
        height=350
    )

    historical_df = pd.read_csv(HISTORICAL_DATA_PATH)
    historical_df['year'] = pd.to_datetime(historical_df['date']).dt.year
    historical_df['month'] = pd.to_datetime(historical_df['date']).dt.month
    yearly_counts = historical_df.groupby('year').size().reset_index(name='count')
    severity_counts = historical_df['severity'].value_counts().reset_index()
    severity_counts.columns = ['severity', 'count']
    monthly_counts = historical_df.groupby('month').size().reset_index(name='count')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_counts['month_name'] = monthly_counts['month'].apply(lambda x: month_names[x-1])

    fig12 = go.Figure()
    fig12.add_trace(go.Bar(x=yearly_counts['year'], y=yearly_counts['count'],
                        name='Yearly Cloudbursts',
                        marker_color='darkblue'))
    fig12.add_trace(go.Bar(x=monthly_counts['month_name'], y=monthly_counts['count'],
                        name='Monthly Pattern',
                        marker_color='skyblue'))
    fig12.add_trace(go.Pie(labels=severity_counts['severity'],
                         values=severity_counts['count'],
                         name='Severity Distribution',
                         domain={'x': [0.7, 1], 'y': [0, 0.5]}))
    fig12.update_layout(
        title="Historical Cloudburst Analysis",
        xaxis=dict(title="Time Period"),
        yaxis=dict(title="Number of Events"),
        height=400,
        legend=dict(x=0, y=1)
    )

    if regional_data is None:
        cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Ahmedabad', 'Pune', city.title()]
        geo_urls = [f"http://api.openweathermap.org/geo/1.0/direct?q={c}&limit=1&appid={API_KEY}" for c in cities]
        coords = []

        for url in geo_urls:
            try:
                response = requests.get(url, timeout=10).json()
                if response:
                    coords.append((response[0]['lat'], response[0]['lon']))
                else:
                    coords.append((0.0, 0.0))
            except Exception as e:
                print(f"Error fetching coordinates for {url.split('=')[1]}: {e}")
                coords.append((0.0, 0.0))

        risks = [0.75, 0.63, 0.42, 0.58, 0.67, 0.53, 0.49, 0.61, probability]

        regional_data = pd.DataFrame({
            'city': cities,
            'lat': [c[0] for c in coords],
            'lon': [c[1] for c in coords],
            'risk': risks
        })

    fig6 = px.scatter_mapbox(regional_data, lat="lat", lon="lon", color="risk",
                            size="risk", color_continuous_scale=px.colors.sequential.Plasma,
                            size_max=25, zoom=3, height=400,
                            mapbox_style="carto-positron",
                            title="Regional Risk Map",
                            hover_name="city",
                            hover_data={"risk": ":.2f"})
    fig6.update_traces(marker=dict(size=30), selector=dict(type='scattermapbox'))

    m = folium.Map(location=[regional_data.loc[regional_data['city'] == city.title(), 'lat'].values[0],
                  regional_data.loc[regional_data['city'] == city.title(), 'lon'].values[0]],
                  zoom_start=10)
    heat_data = [[row['lat'], row['lon'], row['risk']] for _, row in regional_data.iterrows()]
    HeatMap(heat_data, radius=30).add_to(m)
    shelter_locations = [
        (regional_data.loc[regional_data['city'] == city.title(), 'lat'].values[0] + 0.02,
         regional_data.loc[regional_data['city'] == city.title(), 'lon'].values[0] + 0.03,
         "Emergency Shelter 1"),
        (regional_data.loc[regional_data['city'] == city.title(), 'lat'].values[0] - 0.03,
         regional_data.loc[regional_data['city'] == city.title(), 'lon'].values[0] - 0.02,
         "Emergency Shelter 2"),
        (regional_data.loc[regional_data['city'] == city.title(), 'lat'].values[0] + 0.04,
         regional_data.loc[regional_data['city'] == city.title(), 'lon'].values[0] - 0.04,
         "Hospital")
    ]

    for lat, lon, name in shelter_locations:
        folium.Marker(
            location=[lat, lon],
            popup=name,
            icon=folium.Icon(color='green', icon='home')
        ).add_to(m)

    fig_html = m._repr_html_()

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig_html

# --------------------- Prediction System -----------------
class CloudburstPredictor:
    def __init__(self):
        self.model = create_advanced_model((7, 8))
        self.rf_model = create_rf_model()
        self.alert_system = RiskAlertSystem()
        self.api_key = API_KEY
        if not self.api_key or self.api_key == "your_api_key_here":
            raise ValueError("Invalid API key - get one from OpenWeatherMap")

        try:
            self.model.load_weights(MODEL_PATH)
            print("Loaded existing LSTM model")
        except:
            print("Training new LSTM model...")
            X, y = generate_realistic_data()
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            self.model.fit(
                X, y,
                epochs=20,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )
            self.model.save_weights(MODEL_PATH)

        X, y = generate_realistic_data(500)
        X_flat = X.reshape(X.shape[0], -1)
        self.rf_model.fit(X_flat, y.ravel())

    def predict(self, city):
        try:
            url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={self.api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            weather_json = response.json()

            weather_array, weather_details = process_weather_data(weather_json)
            city_coord = (weather_json['city']['coord']['lat'], weather_json['city']['coord']['lon'])
            stats = calculate_stats_and_trends(weather_details)

            # Calculate base probabilities
            lstm_prob = float(self.model.predict(weather_array)[0][0])
            rf_prob = float(self.rf_model.predict_proba(weather_array.reshape(1, -1))[0][1])

            # Adjust probability based on actual weather factors
            precip_factor = min(1.0, stats['precipitation_max'] / 50.0)
            humidity_factor = min(1.0, (stats['humidity_mean'] - 60) / 40.0)
            pressure_factor = 0.5 - (stats['pressure_mean'] - 1013) / 100

            # Combine factors with model predictions
            adjusted_prob = 0.6 * lstm_prob + 0.3 * rf_prob + 0.1 * (
                precip_factor * 0.5 + humidity_factor * 0.3 + pressure_factor * 0.2
            )

            # Ensure probability is between 0 and 1
            final_prob = max(0.0, min(1.0, adjusted_prob))

            # Determine risk level based on final probability
            if final_prob < 0.3:
                risk_level = "Low"
            elif final_prob < 0.6:
                risk_level = "Moderate"
            elif final_prob < 0.8:
                risk_level = "High"
            else:
                risk_level = "Extreme"

            advice = get_safety_advice(risk_level, final_prob, stats)

            # Generate visualizations
            fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, map_html = create_visualizations(
                weather_details, final_prob, city, stats)

            evacuation_routes = self.alert_system.get_evacuation_routes(*city_coord)
            emergency_contacts = self.alert_system.get_emergency_contacts(city)
            forecast_summary = self.generate_forecast_summary(weather_details, stats, final_prob, risk_level)
            alert_html = self.alert_system.generate_alert_html(risk_level, final_prob, advice)

            # Enhanced key statistics section
            key_stats = f"""
            ## 📊 Comprehensive Weather Statistics

            ### Temperature Analysis
            - Mean: {stats['temp_mean']:.1f}°C (Min: {stats['temp_min']:.1f}°C, Max: {stats['temp_max']:.1f}°C)
                       ### Humidity Analysis
            - Mean: {stats['humidity_mean']:.1f}% (Range: {stats['humidity_min']:.1f}-{stats['humidity_max']:.1f}%)
            - Weekly Trend: {stats['humidity_trend']*7:.2f}%

            ### Pressure Analysis
            - Mean: {stats['pressure_mean']:.1f} hPa
            - Weekly Trend: {stats['pressure_trend']*7:.2f} hPa
            - Pressure Changes: {stats.get('wind_direction_change', 0):.1f}°

            ### Precipitation Analysis
            - Mean: {stats['precipitation_mean']:.2f} mm/day
            - Max: {stats['precipitation_max']:.2f} mm/day
            - Days with rain: {stats['precipitation_days']}/7
            - Heavy rain days: {stats['heavy_rain_days']}/7
            - Intensity: {stats['precipitation_intensity']:.2f} mm/rainy day

            ### Wind Analysis
            - Mean Speed: {stats['wind_speed_mean']:.1f} m/s
            - Max Gust: {stats['wind_gust_max']:.1f} m/s
            - Direction Changes: {stats.get('wind_direction_change', 0):.1f}°

            ### Atmospheric Stability
            - Instability Index: {stats.get('instability_index', 0):.2f}
            - Cloud Cover: {stats['cloud_cover_mean']:.1f}%
            - Visibility: {stats['visibility_mean']:.1f} km
            - UV Index: {stats['uv_index_mean']:.1f}

            ### Model Confidence
            - LSTM Probability: {lstm_prob*100:.1f}%
            - Random Forest Probability: {rf_prob*100:.1f}%
            - Combined Probability: {final_prob*100:.1f}%
            """

            prediction_text = f"""
            ## {'🌧️ Cloudburst Likely' if final_prob > 0.5 else '☀️ Clear'}
            **Probability:** {final_prob*100:.1f}%
            **Risk Level:** {risk_level}
            **Location:** {city.title()}

            ## Weather Summary
            {len([w for w in weather_details if w['precipitation'] > 0.1]) / 7 * 100:.1f}% of days with significant rain.
            Mean temperature: {stats['temp_mean']:.1f}°C

            ## Safety Advice
            {advice}

            ## Emergency Contacts
            - **Disaster Management:** {emergency_contacts['disaster_management']}
            - **Police:** {emergency_contacts['police']}
            - **Ambulance:** {emergency_contacts['ambulance']}
            - **Fire:** {emergency_contacts['fire']}

            {key_stats}
            """

            table_data = [[w['date'], f"{w['temp']:.1f}°C", f"{w['humidity']}%",
                         f"{w['pressure']} hPa", f"{w['precipitation']} mm",
                         f"{w['wind_speed']} m/s", f"{w['cloud_cover']}%",
                         f"{w['uv_index']:.1f}", w['description'].title()]
                        for w in weather_details]

            return {
                'prediction_text': prediction_text,
                'table_data': table_data,
                'figures': (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12),
                'map_html': map_html,
                'alert_html': alert_html,
                'evacuation_routes': evacuation_routes,
                'forecast_summary': forecast_summary,
                'audio_file': "alert.mp3" if risk_level == "Moderate" else None,
                'error': None
            }

        except Exception as e:
            error_msg = f"Error predicting for {city}: {str(e)}"
            print(error_msg)
            return {'error': error_msg}

    def generate_forecast_summary(self, weather_data, stats, probability, risk_level):
        rain_days = sum(1 for w in weather_data if w['precipitation'] > 0.1)
        max_temp = max(w['temp'] for w in weather_data)
        min_temp = min(w['temp'] for w in weather_data)
        avg_humidity = sum(w['humidity'] for w in weather_data) / len(weather_data)

        summary = f"""
        <div class="forecast-summary">
            <h3>7-Day Forecast Summary</h3>
            <p>The forecast for the next week shows temperatures ranging from {min_temp:.1f}°C to {max_temp:.1f}°C
            with an average humidity of {avg_humidity:.1f}%.</p>

            <p>We expect precipitation on {rain_days} out of 7 days, with a cloudburst probability of {probability*100:.1f}%.</p>

            <p>The current risk assessment is <strong>{risk_level}</strong>.</p>

            <h4>Key Indicators:</h4>
            <ul>
        """

        if stats.get('pressure_trend', 0) < -1.0:
            summary += "<li>⚠️ Rapidly falling pressure indicates developing storm systems.</li>"

        if stats.get('precipitation_trend', 0) > 0.5:
            summary += "<li>⚠️ Precipitation is expected to intensify throughout the week.</li>"

        if stats.get('wind_speed_max', 0) > 15:
            summary += f"<li>⚠️ Strong winds up to {stats.get('wind_speed_max', 0):.1f} m/s expected.</li>"

        if stats.get('humidity_mean', 0) > 85:
            summary += "<li>⚠️ Very high humidity levels increase the risk of heavy precipitation.</li>"

        if stats.get('instability_index', 0) > 5:
            summary += "<li>⚠️ Atmospheric instability detected - favorable for storm development.</li>"

        summary += """
            </ul>
        </div>
        """

        return summary

# --------------------- User Interface --------------------
predictor = CloudburstPredictor()

def update_interface(city):
    if not city.strip():
        return (
            gr.update(value="Please enter a city name", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    result = predictor.predict(city)

    if result['error']:
        return (
            gr.update(value=f"Error: {result['error']}", visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

    fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12 = result['figures']

    # Show audio player only if we have an alert
    audio_visible = result.get('audio_file') is not None

    return (
        gr.update(value=result['prediction_text'], visible=True),
        gr.update(value=pd.DataFrame(
            result['table_data'],
            columns=['Date', 'Temp', 'Humidity', 'Pressure', 'Rain', 'Wind', 'Cloud Cover', 'UV Index', 'Conditions']
        ), visible=True),
        gr.update(value=fig1, visible=True),
        gr.update(value=fig2, visible=True),
        gr.update(value=fig3, visible=True),
        gr.update(value=fig4, visible=True),
        gr.update(value=fig5, visible=True),
        gr.update(value=fig6, visible=True),
        gr.update(value=fig7, visible=True),
        gr.update(value=fig8, visible=True),
        gr.update(value=fig9, visible=True),
        gr.update(value=fig10, visible=True),
        gr.update(value=fig11, visible=True),
        gr.update(value=fig12, visible=True),
        gr.update(value=result['alert_html'], visible=True),
        gr.update(value=result['map_html'], visible=True),
        gr.update(value=pd.DataFrame(result['evacuation_routes']), visible=True),
        gr.update(value=result['forecast_summary'], visible=True),
        gr.update(value=result.get('audio_file'), visible=audio_visible)
    )

with gr.Blocks(theme=gr.themes.Soft(), title="Advanced Cloudburst Prediction System") as interface:
    gr.Markdown("""
        # 🌩️ Advanced Cloudburst Prediction System
        ## Real-time cloudburst predictions with comprehensive weather analysis and emergency alerts
    """)

    with gr.Row():
        with gr.Column(scale=1):
            city_input = gr.Textbox(label="Enter City", placeholder="e.g., Mumbai", lines=1)
            submit_btn = gr.Button("Predict Cloudburst", variant="primary")

        with gr.Column(scale=2):
            result_output = gr.Markdown(visible=True)
            alert_html = gr.HTML(visible=True, label="Risk Alert")

    with gr.Tabs():
        with gr.TabItem("Basic Weather Data"):
            with gr.Row():
                with gr.Column():
                    chart1_output = gr.Plot(label="Temperature & Precipitation", visible=True)
                    chart2_output = gr.Plot(label="Humidity & Pressure", visible=True)
                    chart4_output = gr.Plot(label="Wind Speed Analysis", visible=True)
                with gr.Column():
                    chart3_output = gr.Plot(label="Cloudburst Probability", visible=True)
                    chart5_output = gr.Plot(label="Weather Conditions Distribution", visible=True)
                    chart6_output = gr.Plot(label="Regional Risk Map", visible=True)

        with gr.TabItem("Advanced Analysis"):
            with gr.Row():
                with gr.Column():
                    chart7_output = gr.Plot(label="Actual vs Feels Like Temperature", visible=True)
                    chart9_output = gr.Plot(label="Heat Index", visible=True)
                    chart11_output = gr.Plot(label="Weather Parameters Radar Analysis", visible=True)
                with gr.Column():
                    chart8_output = gr.Plot(label="Wind Rose Chart", visible=True)
                    chart10_output = gr.Plot(label="Pressure Trend Analysis", visible=True)
                    chart12_output = gr.Plot(label="Historical Cloudburst Analysis", visible=True)

        with gr.TabItem("Emergency Information"):
            with gr.Row():
                with gr.Column():
                    map_html = gr.HTML(label="Evacuation Map with Shelters", visible=True)
                    forecast_summary = gr.HTML(label="Forecast Summary", visible=True)
                with gr.Column():
                    evac_routes = gr.Dataframe(
                        headers=['Route Name', 'Distance', 'Estimated Time', 'Congestion'],
                        row_count=3,
                        col_count=(4, "fixed"),
                        label="Evacuation Routes",
                        interactive=False,
                        visible=True
                    )

        with gr.TabItem("Detailed Forecast"):
            table_output = gr.Dataframe(
                headers=['Date', 'Temp', 'Humidity', 'Pressure', 'Rain', 'Wind', 'Cloud Cover', 'UV Index', 'Conditions'],
                row_count=7,
                col_count=(9, "fixed"),
                interactive=False,
                visible=True
            )

        with gr.TabItem("Voice Alert"):
            with gr.Column():
                audio_output = gr.Audio(label="Risk Alert Voice Note", visible=False)

    gr.Examples(examples=["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "London", "New York"], inputs=city_input)

    submit_btn.click(
        fn=update_interface,
        inputs=city_input,
        outputs=[
            result_output, table_output,
            chart1_output, chart2_output, chart3_output, chart4_output, chart5_output, chart6_output,
            chart7_output, chart8_output, chart9_output, chart10_output, chart11_output, chart12_output,
            alert_html, map_html, evac_routes, forecast_summary, audio_output
        ]
    )

# --------------------- Execution -------------------------
if __name__ == "__main__":
    print("Testing API connection...")
    test_url = f"http://api.openweathermap.org/data/2.5/weather?q=London&appid={API_KEY}"
    try:
        response = requests.get(test_url, timeout=10)
        response.raise_for_status()
        print(f"API Status: {response.status_code}")

        try:
            import plotly.io as pio
            pio.renderers.default = "colab" if 'google.colab' in sys.modules else "browser"
        except:
            pass

        try:
            interface.launch(
                server_name="0.0.0.0",
                server_port=7860,
                debug=True,
                share=True
            )
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"Port 7860 is in use. Trying with alternate port...")
                interface.launch(server_name="0.0.0.0", server_port=None, debug=True, share=True)
            else:
                raise e
    except requests.exceptions.RequestException as e:
        print(f"API Connection failed: {e}")
        print("Running in offline demo mode with sample data...")
        interface.launch()
    except Exception as e:
        print(f"Launch failed: {str(e)}")
        print("Trying with alternate port...")
        interface.launch(server_name="0.0.0.0", server_port=None, debug=True, share=True)
