import streamlit as st
import numpy as np
import pandas as pd
import osmnx as ox
import plotly.express as px

def gaussian_line_source(Q, u, sigma_y, sigma_z, y, z, H=0):
    term_y = np.exp(-(y**2) / (2 * sigma_y**2)) / (np.sqrt(2 * np.pi) * sigma_y)
    term_z = (np.exp(-((z - H)**2) / (2 * sigma_z**2)) +
              np.exp(-((z + H)**2) / (2 * sigma_z**2))) / (np.sqrt(2 * np.pi) * sigma_z)
    return (Q / u) * term_y * term_z

@st.cache_data(show_spinner=False)
def load_network(place_name):
    G = ox.graph_from_place(place_name, network_type='drive')
    G_proj = ox.project_graph(G)
    edges = ox.graph_to_gdfs(G_proj, nodes=False)
    edges['length_m'] = edges.geometry.length
    edges = edges.to_crs(epsg=4326)
    edges['lat'] = edges.geometry.centroid.y
    edges['lon'] = edges.geometry.centroid.x
    edges = edges.dropna(subset=['lat', 'lon', 'length_m'])
    return edges

st.title("Traffic Emissions & Air Quality Dispersion Simulator")

# Sidebar controls
st.sidebar.header("Location & Fleet Composition")

place = st.sidebar.text_input("Enter place (for road network):", "Cambridge, UK")

st.sidebar.subheader("Baseline Fleet Mix (%)")
baseline_diesel = st.sidebar.slider("Baseline Diesel %", 0, 100, 40)
baseline_petrol = st.sidebar.slider("Baseline Petrol %", 0, 100 - baseline_diesel, 50)
baseline_ev = 100 - baseline_diesel - baseline_petrol

st.sidebar.subheader("Policy Fleet Mix (%)")
policy_diesel = st.sidebar.slider("Policy Diesel %", 0, 100, 20)
policy_petrol = st.sidebar.slider("Policy Petrol %", 0, 100 - policy_diesel, 40)
policy_ev = 100 - policy_diesel - policy_petrol

show_baseline = st.sidebar.checkbox("Show baseline", True)
show_policy = st.sidebar.checkbox("Show policy", True)
use_heatmap = st.sidebar.checkbox("Use dispersion heatmap", True)

# Emission factors in g/km
EF = {'diesel': 0.5, 'petrol': 0.3, 'ev': 0.05}

if place:
    edges = load_network(place)

    # Calculate emissions (g/hr)
    edges['emissions_baseline'] = edges['length_m'] / 1000 * (
        baseline_diesel/100 * EF['diesel'] +
        baseline_petrol/100 * EF['petrol'] +
        baseline_ev/100 * EF['ev']
    ) * 3600

    edges['emissions_policy'] = edges['length_m'] / 1000 * (
        policy_diesel/100 * EF['diesel'] +
        policy_petrol/100 * EF['petrol'] +
        policy_ev/100 * EF['ev']
    ) * 3600

    wind_speed = 2.0
    sigma_y = 20.0
    sigma_z = 5.0
    receptor_height = 1.5
    source_height = 0.5

    edges['Q_baseline'] = edges['emissions_baseline'] / 3600
    edges['Q_policy'] = edges['emissions_policy'] / 3600

    edges['conc_baseline'] = edges['Q_baseline'].apply(
        lambda Q: gaussian_line_source(Q, wind_speed, sigma_y, sigma_z, 0, receptor_height, source_height)
    )
    edges['conc_policy'] = edges['Q_policy'].apply(
        lambda Q: gaussian_line_source(Q, wind_speed, sigma_y, sigma_z, 0, receptor_height, source_height)
    )

    # Get overall min/max for color scaling to keep scales consistent
    all_concs = pd.concat([edges['conc_baseline'], edges['conc_policy']])
    cmin, cmax = all_concs.min(), all_concs.max()

    center_lat = edges['lat'].mean()
    center_lon = edges['lon'].mean()
    map_style = "open-street-map"

    def plot_map(df, title):
        if use_heatmap:
            fig = px.density_map(
                df,
                lat='lat',
                lon='lon',
                z='conc',
                radius=20,
                center=dict(lat=center_lat, lon=center_lon),
                zoom=12,
                map_style=map_style,
                color_continuous_scale="YlOrRd",
                range_color=[cmin, cmax],
                title=title,
                opacity=0.7,
            )
        else:
            fig = px.scatter_map(
                df,
                lat='lat',
                lon='lon',
                color='conc',
                size='conc',
                color_continuous_scale="YlOrRd",
                range_color=[cmin, cmax],
                size_max=15,
                zoom=12,
                center=dict(lat=center_lat, lon=center_lon),
                map_style=map_style,
                title=title,
            )
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        return fig

    col1, col2 = st.columns(2)
    if show_baseline:
        with col1:
            st.plotly_chart(
                plot_map(edges[['lat', 'lon', 'conc_baseline']].rename(columns={'conc_baseline':'conc'}), "Baseline Air Quality (NO₂ Proxy)"),
                use_container_width=True
            )
    if show_policy:
        with col2:
            st.plotly_chart(
                plot_map(edges[['lat', 'lon', 'conc_policy']].rename(columns={'conc_policy':'conc'}), "Policy Air Quality (NO₂ Proxy)"),
                use_container_width=True
            )

    # --- Add line chart of emissions over day with rush hour peaks ---
    st.subheader("Emissions over the day")

    # Define hourly traffic factor (simple rush hour pattern)
    hours = np.arange(24)
    traffic_factor = 0.5 + 0.5 * np.sin((hours - 7) / 24 * 2 * np.pi) # Peak at 7-9 am

    total_baseline = edges['emissions_baseline'].sum() / 3600 # convert back to g/s
    total_policy = edges['emissions_policy'].sum() / 3600

    baseline_hourly = total_baseline * traffic_factor
    policy_hourly = total_policy * traffic_factor

    df_line = pd.DataFrame({
        'Hour': hours,
        'Baseline Emissions (g/s)': baseline_hourly,
        'Policy Emissions (g/s)': policy_hourly
    })

    fig_line = px.line(
        df_line,
        x='Hour',
        y=['Baseline Emissions (g/s)', 'Policy Emissions (g/s)'],
        title="Hourly Emissions Pattern (Rush Hours)",
        labels={"value":"Emissions (g/s)", "Hour":"Hour of Day"},
        template="plotly_white"
    )
    st.plotly_chart(fig_line, use_container_width=True)