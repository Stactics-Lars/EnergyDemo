import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def generate_dynamic_pricing():
    """Generate realistic but random electricity prices for the next 48 hours."""
    return np.round(np.random.uniform(0.1, 0.5, 48), 2)

def reset_home_pricing():
    st.session_state.pricing_home = generate_dynamic_pricing()

def predict_energy_demand(device_usage, weather):
    """Simulate AI-based demand prediction."""
    weather_factor = {"Sunny": 1.0, "Cloudy": 1.2, "Night": 1.5}
    base_demand = device_usage * weather_factor[weather]
    predicted_demand = np.clip(base_demand + np.random.normal(0, 5, 48), 0, 100)
    return predicted_demand

def optimize_energy_home(weather, battery_level, grid_price, device_usage):
    """Enhanced AI logic for home energy optimization using predicted demand."""
    solar_generation = 80 if weather == "Sunny" else 30 if weather == "Cloudy" else 0
    predicted_demand = predict_energy_demand(device_usage, weather)
    
    battery_discharge = np.minimum(predicted_demand, battery_level)
    grid_usage = np.maximum(0, predicted_demand - solar_generation - battery_discharge)
    
    if solar_generation > np.mean(predicted_demand):
        decision = "Charge battery with excess solar energy"
    elif np.mean(battery_discharge) > 20 and grid_price == "High":
        decision = "Use battery power to reduce grid usage"
    else:
        decision = "Buy from grid to maintain energy levels"
    
    return decision, solar_generation, battery_discharge, grid_usage, predicted_demand

# Initialize session state for pricing and battery
if 'pricing_home' not in st.session_state:
    st.session_state.pricing_home = generate_dynamic_pricing()
if 'battery_home' not in st.session_state:
    st.session_state.battery_home = [50] * 48

st.sidebar.title("🔋 AI-Powered Energy Optimization")
page = st.sidebar.radio("Select a scenario:", ["Smart Home Optimization"])

if page == "Smart Home Optimization":
    st.title("🏡 Smart Home Energy Optimization")
    weather = st.selectbox("Weather Conditions", ["Sunny", "Cloudy", "Night"])
    battery_level = st.slider("Battery Charge Level (%)", 0, 100, 50)
    grid_price = st.selectbox("Electricity Price", ["Low", "Medium", "High"])
    device_usage = st.slider("Device Energy Usage (kWh)", 0, 50, 10)
    
    home_decision, solar_generation, battery_discharge, grid_usage, predicted_demand = optimize_energy_home(weather, battery_level, grid_price, device_usage)
    st.markdown(f"**AI Decision:** {home_decision}")
    
    # Data Visualization
    df = pd.DataFrame({
        "Battery Level (%)": st.session_state.battery_home,
        "Electricity Price (€/kWh)": st.session_state.pricing_home,
        "Grid Usage (kWh)": [np.mean(grid_usage)] * 48,
        "Predicted Demand (kWh)": predicted_demand
    }, index=range(1, 49))
    
    st.markdown("### 48-Hour Energy Usage & Pricing Forecast")
    st.line_chart(df)
    
    st.button("Regenerate Prices", on_click=reset_home_pricing)

st.success("AI optimizes energy usage for cost savings and efficiency 🚀")
