import streamlit as st
import osmnx as ox
import networkx as nx
import numpy as np
import plotly.express as px
import pandas as pd

# --- Gaussian dispersion ---
def gaussian_dispersion(Q, u, sigma_y, sigma_z, y, z, H=0):
    term_y = np.exp(-(y**2) / (2 * sigma_y**2)) / (np.sqrt(2*np.pi) * sigma_y)
    term_z = (np.exp(-((z-H)**2) / (2 * sigma_z**2)) +
              np.exp(-((z+H)**2) / (2 * sigma_z**2))) / (np.sqrt(2*np.pi) * sigma_z)
    return (Q / u) * term_y * term_z

# --- Emissions calculation ---
def calc_emissions(G, diesel_pct, petrol_pct, ev_pct, traffic_mult, dispersion):
    diesel_ef = 0.8 # g/km arbitrary
    petrol_ef = 0.5
    ev_ef = 0.0

    emissions = []
    for u, v, data in G.edges(data=True):
        length_km = data['length'] / 1000
        base_traffic = data.get('traffic_vol', 100)
        traffic = base_traffic * traffic_mult

        e_diesel = diesel_pct/100 * diesel_ef * length_km * traffic
        e_petrol = petrol_pct/100 * petrol_ef * length_km * traffic
        e_ev = ev_pct/100 * ev_ef * length_km * traffic

        total_emissions = e_diesel + e_petrol + e_ev

        if dispersion:
            total_emissions = gaussian_dispersion(total_emissions, u=2.0, sigma_y=5, sigma_z=2, y=0, z=1.5)

        emissions.append(total_emissions)

    return np.array(emissions)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Traffic Emissions & Dispersion Scenario Explorer")

with st.sidebar:
    st.header("Baseline scenario")
    base_diesel = st.slider("Baseline Diesel %", 0, 100, 50)
    base_petrol = st.slider("Baseline Petrol %", 0, 100, 40)
    base_ev = st.slider("Baseline Electric %", 0, 100, 10)
    base_mult = st.slider("Baseline Traffic Multiplier", 0.1, 3.0, 1.0, 0.1)

    st.header("Policy scenario")
    pol_diesel = st.slider("Policy Diesel %", 0, 100, 30)
    pol_petrol = st.slider("Policy Petrol %", 0, 100, 50)
    pol_ev = st.slider("Policy Electric %", 0, 100, 20)
    pol_mult = st.slider("Policy Traffic Multiplier", 0.1, 3.0, 1.0, 0.1)

    dispersion_toggle = st.checkbox("Apply Gaussian Dispersion", value=False)

# --- Load OSM data ---
place = "Cambridge, UK"
G = ox.graph_from_place(place, network_type="drive")
G = ox.project_graph(G)
#G = ox.add_edge_lengths(G)

# Dummy traffic volumes
for u, v, data in G.edges(data=True):
    data['traffic_vol'] = np.random.randint(50, 200)

# --- Calculate emissions ---
em_base = calc_emissions(G, base_diesel, base_petrol, base_ev, base_mult, dispersion_toggle)
em_pol = calc_emissions(G, pol_diesel, pol_petrol, pol_ev, pol_mult, dispersion_toggle)

# Normalize colors
vmax = max(em_base.max(), em_pol.max())

# --- Convert to GeoDataFrame ---
edges = ox.graph_to_gdfs(G, nodes=False)
edges_base = edges.copy()
edges_base['emissions'] = em_base
edges_pol = edges.copy()
edges_pol['emissions'] = em_pol

# --- Plotting ---
fig_base = px.line_mapbox(edges_base, geojson=edges_base.geometry, color="emissions",
                          color_continuous_scale="YlOrRd", range_color=(0, vmax), zoom=12,
                          mapbox_style="carto-positron")
fig_pol = px.line_mapbox(edges_pol, geojson=edges_pol.geometry, color="emissions",
                         color_continuous_scale="YlOrRd", range_color=(0, vmax), zoom=12,
                         mapbox_style="carto-positron")

# --- Layout ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Baseline Emissions")
    st.plotly_chart(fig_base, use_container_width=True)
with col2:
    st.subheader("Policy Emissions")
    st.plotly_chart(fig_pol, use_container_width=True)
