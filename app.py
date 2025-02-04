import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Real electricity prices for each hour (€/kWh)
real_prices = [
    0.16, 0.16, 0.15, 0.15, 0.15, 0.16, 0.19, 0.25, 0.29, 0.22,
    0.18, 0.16, 0.15, 0.13, 0.15, 0.16, 0.19, 0.24, 0.24, 0.24,
    0.21, 0.18, 0.17, 0.17
]

# ------------------------------------------------------------------
# Generate solar power schedule based on weather for a 24-hour period.
# ------------------------------------------------------------------
def generate_solar_schedule(weather):
    base_solar = {
        "Sunny":      [0, 0, 0, 0, 0, 10, 30, 50, 70, 90, 85, 80, 75, 70, 60, 50, 30, 10, 0, 0, 0, 0, 0, 0],
        "Semi-Sunny": [0, 0, 0, 0, 0, 5, 20, 40, 60, 75, 70, 65, 60, 55, 50, 40, 25, 10, 0, 0, 0, 0, 0, 0],
        "Cloudy":     [0, 0, 0, 0, 0, 2, 10, 20, 35, 50, 45, 40, 35, 30, 25, 20, 10, 5, 0, 0, 0, 0, 0, 0]
    }
    return base_solar[weather]

# ------------------------------------------------------------------
# Updated Smart Home: Optimize appliance schedule using new heuristics.
# Now the function accepts the solar_schedule so that “Least Grid Cost”
# prioritizes hours with high solar production.
# ------------------------------------------------------------------
def optimize_appliance_schedule(appliance_hours, optimization_goal, solar_schedule):
    optimal_schedule = [0] * 24
    solar_array = np.array(solar_schedule)
    if optimization_goal == "Least Grid Power":
        # Schedule load during hours with highest solar generation.
        sorted_indices = np.argsort(-solar_array)
        for i in range(appliance_hours):
            optimal_schedule[sorted_indices[i]] = 50  # each active hour uses 50 kWh (example value)
    elif optimization_goal == "Least Grid Cost":
        # Compute an effective cost that reduces the grid price by a fraction of the (normalized) solar.
        max_solar = solar_array.max() if solar_array.max() != 0 else 1
        # Here the weight is chosen so that the maximum solar contribution roughly equals the grid price spread.
        effective_cost = np.array(real_prices) - (solar_array / max_solar) * (max(real_prices) - min(real_prices))
        sorted_indices = np.argsort(effective_cost)
        for i in range(appliance_hours):
            optimal_schedule[sorted_indices[i]] = 50
    return optimal_schedule

# ------------------------------------------------------------------
# Updated Smart Home: Simulate battery usage with daytime pre‐charging.
# Battery level is expressed in percent (max = 100).
# ------------------------------------------------------------------
def simulate_battery_usage(battery_level, solar_schedule, optimized_schedule):
    battery_state = [battery_level]
    grid_usage = []
    total_grid_cost = 0
    
    for hour in range(24):
        solar_input = solar_schedule[hour]
        appliance_demand = optimized_schedule[hour]
        
        # First: use available solar to meet load.
        direct_solar_usage = min(solar_input, appliance_demand)
        excess_solar = max(0, solar_input - direct_solar_usage)
        battery_discharge = max(0, appliance_demand - direct_solar_usage)
        
        # Discharge battery to help meet the remaining demand.
        current_battery = battery_state[-1]
        if current_battery >= battery_discharge:
            new_battery = current_battery - battery_discharge
            grid_used = 0
        else:
            grid_needed = battery_discharge - current_battery
            new_battery = 0
            grid_used = grid_needed
            total_grid_cost += grid_needed * real_prices[hour]
        
        # Charge battery with any excess solar (do not exceed 100%).
        new_battery = min(new_battery + excess_solar, 100)
        
        # ---- Pre-charge step ----
        # During daytime (6:00-18:00) if the battery isn’t full, charge from grid
        # up to a fixed rate (here, 10% per hour).
        if 6 <= hour < 18 and new_battery < 100:
            precharge_amount = min(100 - new_battery, 10)
            new_battery += precharge_amount
            grid_used += precharge_amount
            total_grid_cost += precharge_amount * real_prices[hour]
        
        battery_state.append(new_battery)
        grid_usage.append(grid_used)
    
    return battery_state[:-1], grid_usage, total_grid_cost

# ------------------------------------------------------------------
# Updated Factory Optimization: Schedule production hours.
# Now “Least Grid Cost” uses the effective cost (grid price reduced by solar)
# and “Least Grid Power” schedules production when solar is highest.
# ------------------------------------------------------------------
def optimize_factory_schedule(production_hours, optimization_goal, production_demand, solar_schedule):
    schedule = [0] * 24
    solar_array = np.array(solar_schedule)
    if optimization_goal == "Least Grid Cost":
        max_solar = solar_array.max() if solar_array.max() != 0 else 1
        effective_cost = np.array(real_prices) - (solar_array / max_solar) * (max(real_prices)-min(real_prices))
        sorted_indices = np.argsort(effective_cost)
        for i in range(production_hours):
            schedule[sorted_indices[i]] = production_demand
    elif optimization_goal == "Least Grid Power":
        sorted_indices = np.argsort(-solar_array)
        for i in range(production_hours):
            schedule[sorted_indices[i]] = production_demand
    return schedule

# ------------------------------------------------------------------
# Updated Factory: Simulate energy flow with battery pre‐charging.
# Battery is measured in kWh.
# ------------------------------------------------------------------
def simulate_factory_usage(battery_charge, battery_capacity, solar_schedule, production_schedule):
    battery_states = [battery_charge]
    grid_usage = []
    total_grid_cost = 0
    for hour in range(24):
        solar_input = solar_schedule[hour]
        production_demand = production_schedule[hour]
        
        # Use solar directly for production.
        direct_solar_usage = min(solar_input, production_demand)
        remaining_demand = production_demand - direct_solar_usage
        
        # Use battery to cover remaining demand.
        current_battery = battery_states[-1]
        battery_discharge = min(current_battery, remaining_demand)
        remaining_demand -= battery_discharge
        new_battery = current_battery - battery_discharge
        
        # Charge battery with any excess solar (but not beyond capacity).
        excess_solar = max(0, solar_input - production_demand)
        new_battery = min(new_battery + excess_solar, battery_capacity)
        
        # Any still unmet demand is supplied by the grid.
        grid_used = remaining_demand
        total_grid_cost += grid_used * real_prices[hour]
        
        # ---- Pre-charge step ----
        # During daytime (6:00-18:00), if battery isn’t full, pre-charge from grid
        # at a fixed rate (here, up to 5% of capacity per hour).
        if 6 <= hour < 18 and new_battery < battery_capacity:
            precharge_amount = min(battery_capacity - new_battery, battery_capacity * 0.05)
            new_battery += precharge_amount
            grid_used += precharge_amount
            total_grid_cost += precharge_amount * real_prices[hour]
        
        battery_states.append(new_battery)
        grid_usage.append(grid_used)
    
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
    appliance_usage = st.slider("Appliance Usage Hours", 0, 24, 5)
    optimization_goal = st.selectbox("Optimization Goal", ["Least Grid Power", "Least Grid Cost"])
    
    solar_schedule = generate_solar_schedule(weather)
    optimized_schedule = optimize_appliance_schedule(appliance_usage, optimization_goal, solar_schedule)
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
    }, index=[datetime.time(i, 0).strftime('%H') for i in range(24)])
    
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
    
    st.success("AI optimizes energy usage with solar prioritization and daytime battery pre‐charging!")

# ------------------------------
# Factory Optimization Page
# ------------------------------
elif page == "Factory Optimization":
    st.title("🏭 Factory Energy Optimization Demo")
    st.markdown(
        """
        This demo scales up the simulation to represent a factory’s energy management.
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
    
    st.success("Factory energy optimization demo complete with solar‐focused scheduling and daytime battery pre‐charging!")
