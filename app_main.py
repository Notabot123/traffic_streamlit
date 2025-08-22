# streamlit_app_full.py
# Requirements: streamlit, osmnx, plotly, numpy, pandas, scipy

import streamlit as st
import osmnx as ox
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.ndimage import gaussian_filter

st.set_page_config(layout="wide", page_title="Traffic → Emissions → Dispersion Demo")

########################
# Helper functions
########################

def time_of_day_multiplier(hour):
    """Synthetic multi-modal daily traffic multiplier curve (normalized).
       Two Gaussian peaks near 09:00 and 17:00 plus a base."""
    h = np.arange(24)
    morning = np.exp(-0.5 * ((h - 9) / 1.5) ** 2)
    evening = np.exp(-0.5 * ((h - 17) / 1.75) ** 2)
    base = 0.2 + 0.05 * np.cos((h - 13) * 2 * np.pi / 24)
    curve = base + 0.9 * morning + 0.9 * evening
    # normalize so the mean is 1.0 (so multipliers scale sensibly)
    return curve / np.mean(curve)

def mm1_mean_delay(lambda_rate, mu_rate):
    """Analytical M/M/1 mean waiting time in system (W = 1/(mu-lambda) if lambda<mu), returns seconds.
       lambda_rate and mu_rate in customers/sec. If lambda>=mu, return a large capped value."""
    if lambda_rate <= 0:
        return 0.0
    if lambda_rate >= mu_rate:
        return 3600.0 # cap huge delays at 1 hour for stability
    W = 1.0 / (mu_rate - lambda_rate) # seconds (since rates are per second)
    return W

def gaussian_line_concentration(Q_g_per_s, u_m_s, sigma_y, sigma_z, y, z, H=0.0):
    """Simplified Gaussian line source at receptor (y lateral offset, z height).
       Returns concentration in g/m3. (Single-receptor simplistic)"""
    term_y = np.exp(-(y**2) / (2 * sigma_y**2)) / (np.sqrt(2*np.pi) * sigma_y)
    term_z = (np.exp(-((z - H)**2) / (2 * sigma_z**2)) +
              np.exp(-((z + H)**2) / (2 * sigma_z**2))) / (np.sqrt(2*np.pi) * sigma_z)
    C = (Q_g_per_s / (u_m_s + 1e-9)) * term_y * term_z
    return C

def create_dispersion_grid(points_df, val_col, grid_size=200, sigma=3):
    """Fast grid-based smear of point emissions: bin to grid then gaussian blur."""
    lat_min, lat_max = points_df['lat'].min(), points_df['lat'].max()
    lon_min, lon_max = points_df['lon'].min(), points_df['lon'].max()
    # create bins
    lat_bins = np.linspace(lat_min, lat_max, grid_size)
    lon_bins = np.linspace(lon_min, lon_max, grid_size)
    grid = np.zeros((grid_size, grid_size))
    lat_idx = np.searchsorted(lat_bins, points_df['lat']) - 1
    lon_idx = np.searchsorted(lon_bins, points_df['lon']) - 1
    for i, (li, lj, v) in enumerate(zip(lat_idx, lon_idx, points_df[val_col])):
        if 0 <= li < grid_size and 0 <= lj < grid_size:
            grid[li, lj] += v
    # gaussian blur
    grid = gaussian_filter(grid, sigma=sigma)
    return lat_bins, lon_bins, grid

########################
# Sidebar controls
########################
st.sidebar.title("Scenario Controls")

st.sidebar.header("Location")
place = st.sidebar.text_input("Place (OSM query)", "Cambridge, UK")

st.sidebar.header("Baseline fleet (percentages must sum ≤ 100)")
base_ev = st.sidebar.slider("Baseline EV %", 0, 100, 10)
base_diesel = st.sidebar.slider("Baseline Diesel %", 0, 100, 40)
base_petrol = st.sidebar.slider("Baseline Petrol %", 0, 100, 50)
# note: we do not auto-balance; user controls values

st.sidebar.header("Policy fleet (percentages must sum ≤ 100)")
pol_ev = st.sidebar.slider("Policy EV %", 0, 100, 30)
pol_diesel = st.sidebar.slider("Policy Diesel %", 0, 100, 30)
pol_petrol = st.sidebar.slider("Policy Petrol %", 0, 100, 40)

st.sidebar.header("Traffic multipliers")
base_mult_global = st.sidebar.slider("Baseline traffic multiplier (global)", 0.1, 3.0, 1.0, 0.1)
pol_mult_global = st.sidebar.slider("Policy traffic multiplier (global)", 0.1, 3.0, 1.0, 0.1)

st.sidebar.header("Queue (congestion) model")
queue_enabled = st.sidebar.checkbox("Enable queue delays (analytical M/M/1)", value=False)
# arrival/service rates are per-second base; scaled by traffic multipliers and time-of-day
arrival_rate_per_min = st.sidebar.slider("Base arrival rate (vehicles/min) on major roads", 30, 600, 120)
service_rate_per_min = st.sidebar.slider("Service rate (vehicles/min) capacity per major road", 60, 1000, 240)
queue_show_heat = st.sidebar.checkbox("Show queue delay heatmap (instead of emissions)", value=False)

st.sidebar.header("Dispersion & plotting")
apply_dispersion = st.sidebar.checkbox("Apply Gaussian dispersion to emissions (heatmap)", value=False)
use_heatmap = st.sidebar.checkbox("Use density heatmap plotting (vs points)", value=True)

st.sidebar.header("Time of day")
hour = st.sidebar.slider("Hour of day", 0, 23, 9)
st.sidebar.markdown("Traffic curve has peaks around 09:00 and 17:00 (synthetic)")

########################
# Load network (cached)
########################
@st.cache_data(show_spinner=False)
def load_osm_graph(place_name):
    try:
        G = ox.graph_from_place(place_name, network_type="drive")
    except:
        # save graph
        ox.load_graphml(G,"cambridge.graphml")
        st.warning(f"Unable to connect to load {place_name}. Default to saved road dave for Cambridge.")
    # project to metric system for accurate lengths and speeds
    Gp = ox.project_graph(G)
    # attach length and default speed where available
    for u, v, k, data in Gp.edges(keys=True, data=True):
        # length exists in geometry; keep in meters
        if 'length' not in data:
            data['length'] = 0.0
        # if maxspeed present use it, else fallback to 50 km/h
        if 'maxspeed' in data:
            try:
                if isinstance(data['maxspeed'], list):
                    ms = data['maxspeed'][0]
                else:
                    ms = data['maxspeed']
                # extract numeric (some are '30 mph' or '50')
                ms_num = float(str(ms).split()[0])
                # units: if in mph assume mph else assume km/h; naive approach
                if 'mph' in str(ms):
                    # convert mph to km/h
                    ms_num = ms_num * 1.60934
                data['speed_kph'] = ms_num
            except Exception:
                data['speed_kph'] = 50.0
        else:
            data['speed_kph'] = 50.0
        # classify major roads flag
        hw = data.get('highway', '')
        if isinstance(hw, list):
            hw = hw[0]
        data['is_major'] = hw in ('motorway', 'trunk', 'primary')
        # default base traffic volume (veh/hr) if not provided
        data.setdefault('base_veh_hr', np.random.randint(50, 250))
    return Gp

G = load_osm_graph(place)

# quick guard: ensure graph has edges
if len(list(G.edges(keys=True))) == 0:
    st.error("No edges returned for that place. Try a larger town/city name e.g. 'Cambridge, UK' or 'Bristol, UK'.")
    st.stop()

########################
# Prepare edges dataframe
########################
edges_gdf = ox.graph_to_gdfs(G, nodes=False)
# reproject back to WGS84 for plotting lat/lon
edges_wgs = edges_gdf.to_crs(epsg=4326).copy()
edges_wgs['lat'] = edges_wgs.geometry.centroid.y
edges_wgs['lon'] = edges_wgs.geometry.centroid.x

# basic emission factors (g/km) for an NO2-like proxy (simple placeholders)
EF = {
    'diesel': 0.6, # g/km
    'petrol': 0.35,
    'ev': 0.02 # small non-tailpipe proxy
}

# prepare synthetic daily traffic multiplier curve
tod_curve = time_of_day_multiplier(hour=hour) # returns whole curve but we will index hour
tod_full = time_of_day_multiplier(0) # actually returns full array; but function returns array ignoring arg
# However the function signature was written to output the full curve; fix:
tod_array = time_of_day_multiplier(0) # full 24-length array
tod_current = tod_array[hour]

# compute baseline and policy emissions per edge, factoring fleet mix, traffic multipliers, and time-of-day
def compute_edge_emissions(edges_df, diesel_pct, petrol_pct, ev_pct, traffic_mult, tod_factor,
                           apply_queue=False, arrival_min=120, service_min=240):
    # edges_df is Gp graph via graph_to_gdfs projection aligned with metrics (we used projected earlier)
    # We'll calculate for each edge:
    emissions_list = []
    delays_seconds = []
    speeds_kph = []
    for idx, row in edges_df.iterrows():
        length_m = row.get('length', row.geometry.length) # meters
        length_km = float(length_m) / 1000.0
        base_veh_hr = float(row.get('base_veh_hr', 100.0))
        veh_hr = base_veh_hr * traffic_mult * tod_factor
        # basic emissions (g/hr) ignoring speed changes:
        e_diesel = diesel_pct / 100.0 * EF['diesel'] * length_km * veh_hr
        e_petrol = petrol_pct / 100.0 * EF['petrol'] * length_km * veh_hr
        e_ev = ev_pct / 100.0 * EF['ev'] * length_km * veh_hr
        total_emissions_g_hr = e_diesel + e_petrol + e_ev

        # queue/delay (analytical M/M/1) for major roads if enabled
        delay_s = 0.0
        if apply_queue and bool(row.get('is_major', False)):
            # arrival and service rates in veh/sec (global base scaled by veh_hr)
            lam = (arrival_min / 60.0) * (veh_hr / max(1.0, base_veh_hr)) # scale arrival by relative traffic
            mu = service_min / 60.0 # service capacity in vehicles/sec (global)
            delay_s = mm1_mean_delay(lam, mu)
        # compute free-flow travel time (s) and reduced by delay
        speed_kph = float(row.get('speed_kph', 50.0))
        # travel time seconds for edge
        travel_time_s = (length_km / max(0.001, speed_kph / 3.6))
        # effective travel time:
        eff_travel_time_s = travel_time_s + delay_s
        # effective speed in kph (avoid zero)
        eff_speed_kph = max(1.0, length_km / (eff_travel_time_s / 3600.0))
        # consider speed effect on emissions: simple scaling factor (higher emissions at lower speed)
        # scale = reference_speed / eff_speed
        scale_speed = (speed_kph / eff_speed_kph)
        scale_speed = max(0.5, min(scale_speed, 3.0)) # cap scaling to avoid extremes
        total_emissions_g_hr *= scale_speed

        emissions_list.append(total_emissions_g_hr)
        delays_seconds.append(delay_s)
        speeds_kph.append(eff_speed_kph)

    return np.array(emissions_list), np.array(delays_seconds), np.array(speeds_kph)

# compute baseline & policy
em_base, delay_base, speed_base = compute_edge_emissions(
    edges_gdf, base_diesel, base_petrol, base_ev, base_mult_global, tod_current,
    apply_queue=queue_enabled, arrival_min=arrival_rate_per_min, service_min=service_rate_per_min
)
em_pol, delay_pol, speed_pol = compute_edge_emissions(
    edges_gdf, pol_diesel, pol_petrol, pol_ev, pol_mult_global, tod_current,
    apply_queue=queue_enabled, arrival_min=arrival_rate_per_min, service_min=service_rate_per_min
)

# attach to plotting dataframe
edges_wgs['em_base'] = em_base
edges_wgs['em_pol'] = em_pol
edges_wgs['delay_base_s'] = delay_base
edges_wgs['delay_pol_s'] = delay_pol
edges_wgs['speed_base_kph'] = speed_base
edges_wgs['speed_pol_kph'] = speed_pol

# choose which column to plot
if queue_show_heat:
    plot_col_base = 'delay_base_s'
    plot_col_pol = 'delay_pol_s'
    colorbar_title = "Delay (s)"
else:
    # when plotting emissions, if dispersion selected compute concentrations using gaussian approx
    if apply_dispersion:
        # convert edge emissions g/hr -> g/s Q and compute simple line receptor concentration proxy
        u = 2.0
        sigma_y = 20.0
        sigma_z = 5.0
        receptor_z = 1.5
        edge_Q_base = edges_wgs['em_base'].values / 3600.0
        edge_Q_pol = edges_wgs['em_pol'].values / 3600.0
        conc_base = np.array([gaussian_line_concentration(Q, u, sigma_y, sigma_z, 0.0, receptor_z, H=0.5)
                              for Q in edge_Q_base])
        conc_pol = np.array([gaussian_line_concentration(Q, u, sigma_y, sigma_z, 0.0, receptor_z, H=0.5)
                             for Q in edge_Q_pol])
        edges_wgs['conc_base'] = conc_base
        edges_wgs['conc_pol'] = conc_pol
        plot_col_base = 'conc_base'
        plot_col_pol = 'conc_pol'
        colorbar_title = "Conc (g/m³) (proxy)"
    else:
        plot_col_base = 'em_base'
        plot_col_pol = 'em_pol'
        colorbar_title = "Emissions (g/hr)"

# shared color scale
combined_vals = np.concatenate([edges_wgs[plot_col_base].values, edges_wgs[plot_col_pol].values])
cmin = float(np.nanpercentile(combined_vals, 1))
cmax = float(np.nanpercentile(combined_vals, 99))

########################
# Plot maps side by side
########################
st.title("Baseline vs Policy — Emissions / Delay Maps")

col1, col2 = st.columns(2)
map_style = "open-street-map"

if use_heatmap:
    # prepare points dataframe for density map: use centroids and the relevant column
    df_base_pts = edges_wgs[['lat', 'lon', plot_col_base]].rename(columns={plot_col_base: 'val'}).copy()
    df_pol_pts = edges_wgs[['lat', 'lon', plot_col_pol]].rename(columns={plot_col_pol: 'val'}).copy()

    with col1:
        st.subheader("Baseline")
        fig_b = px.density_map(
            df_base_pts, lat='lat', lon='lon', z='val', radius=20,
            center=dict(lat=df_base_pts['lat'].mean(), lon=df_base_pts['lon'].mean()),
            zoom=12, map_style=map_style, color_continuous_scale="YlOrRd",
            range_color=(cmin, cmax), title="Baseline"
        )
        fig_b.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_b, use_container_width=True)

    with col2:
        st.subheader("Policy")
        fig_p = px.density_map(
            df_pol_pts, lat='lat', lon='lon', z='val', radius=20,
            center=dict(lat=df_pol_pts['lat'].mean(), lon=df_pol_pts['lon'].mean()),
            zoom=12, map_style=map_style, color_continuous_scale="YlOrRd",
            range_color=(cmin, cmax), title="Policy"
        )
        fig_p.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_p, use_container_width=True)
else:
    # scatter / line plotting option: show edges colored by value; using scatter_map for speed
    df_base_pts = edges_wgs[['lat', 'lon', plot_col_base]].rename(columns={plot_col_base: 'val'}).copy()
    df_pol_pts = edges_wgs[['lat', 'lon', plot_col_pol]].rename(columns={plot_col_pol: 'val'}).copy()

    with col1:
        st.subheader("Baseline")
        fig_b = px.scatter_map(df_base_pts, lat='lat', lon='lon', color='val',
                               color_continuous_scale="YlOrRd", range_color=(cmin, cmax),
                               center=dict(lat=df_base_pts['lat'].mean(), lon=df_base_pts['lon'].mean()),
                               zoom=12, map_style=map_style, title="Baseline")
        fig_b.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_b, use_container_width=True)

    with col2:
        st.subheader("Policy")
        fig_p = px.scatter_map(df_pol_pts, lat='lat', lon='lon', color='val',
                               color_continuous_scale="YlOrRd", range_color=(cmin, cmax),
                               center=dict(lat=df_pol_pts['lat'].mean(), lon=df_pol_pts['lon'].mean()),
                               zoom=12, map_style=map_style, title="Policy")
        fig_p.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        st.plotly_chart(fig_p, use_container_width=True)

########################
# Hourly emissions plot (24h synthetic)
########################
st.markdown("---")
st.subheader("24-hour Emissions Profile (synthetic)")

hours = np.arange(24)
tod_curve = time_of_day_multiplier(0) # returns whole curve of length 24
# compute total emissions per hour for baseline & policy by scaling totals with tod curve
total_em_base = edges_wgs['em_base'].sum()
total_em_pol = edges_wgs['em_pol'].sum()

hourly_base = total_em_base * tod_curve
hourly_pol = total_em_pol * tod_curve

df_hours = pd.DataFrame({
    'hour': hours,
    'Baseline (g/hr)': hourly_base,
    'Policy (g/hr)': hourly_pol
})

fig_line = px.line(df_hours, x='hour', y=['Baseline (g/hr)', 'Policy (g/hr)'],
                   labels={'value': 'Total emissions (g/hr)', 'hour': 'Hour of day'},
                   title="Synthetic 24-hour Emissions (peaks at ~09:00 & ~17:00)")
# add vertical marker for current hour
fig_line.add_vline(x=hour, line_dash="dash", line_color="black")

st.plotly_chart(fig_line, use_container_width=True)

st.markdown("**Notes & next steps:** This demo uses a simple analytical M/M/1 queue for major roads (fast). "
            "For more realism you can: (a) swap to short SimPy runs per-edge; (b) ingest DfT/TomTom hourly counts; "
            "(c) swap simplified EF and dispersion for ADMS/CFD outputs where high fidelity required.")