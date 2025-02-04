import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Real electricity prices from today (€/kWh)
real_prices = [
    0.16, 0.16, 0.15, 0.15, 0.15, 0.16, 0.19, 0.25, 0.29, 0.22,
    0.18, 0.16, 0.15, 0.13, 0.15, 0.16, 0.19, 0.24, 0.24, 0.24,
    0.21, 0.18, 0.17, 0.17
]

# Generate solar power schedule based on weather for a 24-hour period
def generate_solar_schedule(weather):
    base_solar = {
        "Sunny":      [0, 0, 0, 0, 0, 10, 30, 50, 70, 90, 85, 80, 75, 70, 60, 50, 30, 10, 0, 0, 0, 0, 0, 0],
        "Semi-Sunny": [0, 0, 0, 0, 0, 5, 20, 40, 60, 75, 70, 65, 60, 55, 50, 40, 25, 10, 0, 0, 0, 0, 0, 0],
        "Cloudy":     [0, 0, 0, 0, 0, 2, 10, 20, 35, 50, 45, 40, 35, 30, 25, 20, 10, 5, 0, 0, 0, 0, 0, 0]
    }
    return base_solar[weather]

# Smart Home: Optimize appliance schedule (using a simple heuristic)
def optimize_appliance_schedule(appliance_consumption, optimization_goal):
    sorted_indices = np.argsort(real_prices)
    optimal_schedule = [0] * 24
    
    if optimization_goal == "Least Grid Power":
        # A simple heuristic: spread appliance use evenly.
        for i in range(appliance_consumption):
            optimal_schedule[i] = 50  # each active hour uses 50 kWh (example value)
    elif optimization_goal == "Least Grid Cost":
        # Schedule appliances during the cheapest hours.
        for idx in sorted_indices[:appliance_consumption]:
            optimal_schedule[idx] = 50
    return optimal_schedule

# Smart Home: Simulate battery usage for a 24-hour cycle.
def simulate_battery_usage(battery_level, solar_schedule, optimized_schedule):
    battery_state = [battery_level]
    grid_usage = []
    total_grid_cost = 0
    
    for hour in range(24):
        solar_input = solar_schedule[hour]
        appliance_demand = optimized_schedule[hour]
        # Use available solar to cover appliance demand first.
        direct_solar_usage = min(solar_input, appliance_demand)
        excess_solar = max(0, solar_input - direct_solar_usage)
        battery_discharge = max(0, appliance_demand - direct_solar_usage)
        
        # Charge battery with any excess solar (capped at 100%)
        if battery_state[-1] + excess_solar <= 100:
            battery_state.append(battery_state[-1] + excess_solar)
        else:
            battery_state.append(100)
        
        # Discharge battery if needed; otherwise, draw from the grid.
        if battery_discharge > 0:
            if battery_state[-1] >= battery_discharge:
                battery_state[-1] -= battery_discharge
                grid_usage.append(0)
            else:
                grid_needed = battery_discharge - battery_state[-1]
                grid_usage.append(grid_needed)
                battery_state[-1] = 0
                total_grid_cost += grid_needed * real_prices[hour]
        else:
            grid_usage.append(0)
    
    return battery_state[:-1], grid_usage, total_grid_cost

# ============================================================================
# New functions for the Factory (Scaled) Optimization Demo
# ============================================================================

def optimize_factory_schedule(production_hours, optimization_goal, production_demand, solar_schedule):
    """
    Schedules the factory production hours over a day.
    
    - For "Least Grid Cost": production is scheduled during the hours with the lowest electricity prices.
    - For "Least Grid Power": production is scheduled during the hours with the highest solar generation.
    """
    schedule = [0] * 24
    if optimization_goal == "Least Grid Cost":
        sorted_indices = np.argsort(real_prices)
        for i in range(production_hours):
            schedule[sorted_indices[i]] = production_demand
    elif optimization_goal == "Least Grid Power":
        # Schedule production when solar generation is highest.
        sorted_indices = np.argsort(-np.array(solar_schedule))
        for i in range(production_hours):
            schedule[sorted_indices[i]] = production_demand
    return schedule

def simulate_factory_usage(battery_charge, battery_capacity, solar_schedule, production_schedule):
    """
    Simulates the energy flow for a factory over 24 hours.
    
    For each hour:
      - Solar energy is used directly to meet production demand.
      - Any shortfall is met by discharging the battery.
      - Remaining demand is met from the grid (incurring cost).
      - Excess solar energy charges the battery (capped by battery capacity).
    """
    battery_states = [battery_charge]
    grid_usage = []
    total_grid_cost = 0
    for hour in range(24):
        solar_input = solar_schedule[hour]
        production_demand = production_schedule[hour]
        # Use solar directly for production.
        direct_solar_usage = min(solar_input, production_demand)
        remaining_demand = production_demand - direct_solar_usage
        # Discharge battery to help cover remaining demand.
        battery_discharge = min(battery_states[-1], remaining_demand)
        remaining_demand -= battery_discharge
        # Charge battery with any excess solar.
        excess_solar = max(0, solar_input - production_demand)
        new_battery_charge = battery_states[-1] - battery_discharge + excess_solar
        new_battery_charge = min(new_battery_charge, battery_capacity)
        battery_states.append(new_battery_charge)
        # Any unmet demand is supplied by the grid.
        grid_used = remaining_demand
        grid_usage.append(grid_used)
        total_grid_cost += grid_used * real_prices[hour]
    return battery_states[:-1], grid_usage, total_grid_cost

# ============================================================================
# Streamlit App Layout with Two Pages: Smart Home & Factory Optimization
# ============================================================================

st.sidebar.title("🔋 AI-Powered Energy Optimization")
st.sidebar.markdown("Choose your scenario and settings.")
page = st.sidebar.radio("Select a scenario:", ["Smart Home Optimization", "Factory Optimization"])

# ------------------------------
# Smart Home Optimization Page
# ------------------------------
if page == "Smart Home Optimization":
    st.title("🏡 Smart Home Energy Optimization")
    weather = st.selectbox("Weather Conditions", ["Sunny", "Semi-Sunny", "Cloudy"])
    battery_level = st.slider("Battery Charge Level (%)", 0, 100, 50)
    appliance_usage = st.slider("Appliance Energy Usage (hours)", 0, 24, 5)
    optimization_goal = st.selectbox("Optimization Goal", ["Least Grid Power", "Least Grid Cost"])
    
    solar_schedule = generate_solar_schedule(weather)
    optimized_schedule = optimize_appliance_schedule(appliance_usage, optimization_goal)
    battery_state, grid_usage, total_grid_cost = simulate_battery_usage(battery_level, solar_schedule, optimized_schedule)
    
    st.markdown("### Optimized Appliance Schedule")
    active_hours = [str(i) for i, v in enumerate(optimized_schedule) if v == 50]
    st.markdown(f"**Appliance should run during hours:** {', '.join(active_hours)}")
    st.markdown(f"**Total Grid Cost (€):** {total_grid_cost:.2f}")
    
    # Visualization for Smart Home
    df = pd.DataFrame({
        "Electricity Price (€/kWh)": real_prices,
        "Solar Generation (kWh)": solar_schedule,
        "Appliance Schedule (1=ON)": optimized_schedule,
        "Battery Level (%)": battery_state,
        "Grid Usage (kWh)": grid_usage
    }, index=[datetime.time(i, 0).strftime('%H:%M') for i in range(24)])
    
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()
    ax1.plot(df.index, df["Electricity Price (€/kWh)"], 'g-', label='Electricity Price')
    ax2.plot(df.index, df["Solar Generation (kWh)"], 'b-', label='Solar Generation')
    ax2.plot(df.index, df["Appliance Schedule (1=ON)"], 'r-', label='Appliance Usage')
    ax2.plot(df.index, df["Battery Level (%)"], 'orange', label='Battery Level')
    ax2.plot(df.index, df["Grid Usage (kWh)"], 'purple', label='Grid Usage')
    
    ax1.set_xlabel("Time (Hours)")
    ax1.set_ylabel("Price (€/kWh)", color='g')
    ax2.set_ylabel("Energy (kWh/%)", color='b')
    
    ax1.set_ylim(0, max(real_prices)*1.2)
    ax2.set_ylim(0, 110)
    
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    st.pyplot(fig)
    
    st.success("AI optimizes energy usage for cost savings and efficiency 🚀")

# ------------------------------
# Factory Optimization Page
# ------------------------------
elif page == "Factory Optimization":
    st.title("🏭 Factory Energy Optimization Demo")
    st.markdown(
        """
        This demo scales up the simulation to represent a factory’s energy management.
        Adjust the parameters below to see how production operations can be scheduled
        for cost savings while taking advantage of on-site solar generation and a large battery system.
        """
    )
    
    weather = st.selectbox("Weather Conditions", ["Sunny", "Semi-Sunny", "Cloudy"], key="factory_weather")
    battery_capacity = st.slider("Factory Battery Capacity (kWh)", 100, 2000, 1000, step=50)
    battery_charge = st.slider("Current Battery Charge (kWh)", 0, battery_capacity, battery_capacity // 2, step=10)
    production_hours = st.slider("Production Operation Hours", 0, 24, 12)
    production_demand = st.slider("Production Energy per Hour (kWh)", 50, 1000, 200, step=10)
    optimization_goal = st.selectbox("Optimization Goal", ["Least Grid Power", "Least Grid Cost"], key="factory_goal")
    solar_scale = st.slider("Factory Solar Scaling Factor", 1, 20, 10)
    
    # Generate a scaled solar schedule for the factory.
    base_solar = generate_solar_schedule(weather)
    factory_solar_schedule = [val * solar_scale for val in base_solar]
    
    # Schedule production hours based on the selected optimization goal.
    production_schedule = optimize_factory_schedule(production_hours, optimization_goal, production_demand, factory_solar_schedule)
    
    battery_states, grid_usage, total_grid_cost = simulate_factory_usage(battery_charge, battery_capacity, factory_solar_schedule, production_schedule)
    
    st.markdown("### Optimized Production Schedule")
    prod_hours = [str(i) for i, v in enumerate(production_schedule) if v > 0]
    st.markdown(f"**Production should run during hours:** {', '.join(prod_hours)}")
    st.markdown(f"**Total Grid Cost (€):** {total_grid_cost:.2f}")
    
    # Visualization for Factory Optimization
    df_factory = pd.DataFrame({
        "Electricity Price (€/kWh)": real_prices,
        "Factory Solar Generation (kWh)": factory_solar_schedule,
        "Production Schedule (kWh)": production_schedule,
        "Battery Level (kWh)": battery_states,
        "Grid Usage (kWh)": grid_usage
    }, index=[datetime.time(i, 0).strftime('%H:%M') for i in range(24)])
    
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax2 = ax1.twinx()
    ax1.plot(df_factory.index, df_factory["Electricity Price (€/kWh)"], 'g-', label='Electricity Price')
    ax2.plot(df_factory.index, df_factory["Factory Solar Generation (kWh)"], 'b-', label='Solar Generation')
    ax2.plot(df_factory.index, df_factory["Production Schedule (kWh)"], 'r-', label='Production Demand')
    ax2.plot(df_factory.index, df_factory["Battery Level (kWh)"], 'orange', label='Battery Level')
    ax2.plot(df_factory.index, df_factory["Grid Usage (kWh)"], 'purple', label='Grid Usage')
    
    ax1.set_xlabel("Time (Hours)")
    ax1.set_ylabel("Price (€/kWh)", color='g')
    ax2.set_ylabel("Energy (kWh)", color='b')
    
    ax1.set_ylim(0, max(real_prices)*1.2)
    ax2.set_ylim(0, battery_capacity * 1.1)
    
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
    st.pyplot(fig)
    
    st.success("Factory energy optimization demo complete. Optimize your production schedule for maximum savings!")
