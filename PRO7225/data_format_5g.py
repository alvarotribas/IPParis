"""
PIR - Projet d'Initiation a la recherche @ Telecom Paris
Code 01 - Processing of the data from the Cartoradio in the Paris region
"""

# 0 - Imports ===========================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import osmnx as ox
import os
import glob

# 1 - Opening map files, display and plot ===============================================================================================

# Opening "Sites_Cartoradio" file from pc directory 
df_sites = pd.read_csv('~/Documents/University/TelecomSudParis/PIR/cartoradio_ParisArea_06-11-25/Sites_Cartoradio.csv',
                 delimiter=';', encoding='latin1')

# Converting to library frame
gdf = gpd.GeoDataFrame(
    df_sites, geometry=gpd.points_from_xy(df_sites.Longitude, df_sites.Latitude), crs="EPSG:4326"
)

# Downloading street network and converting its graph
G = ox.graph_from_place('Paris, France', network_type='drive')
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
edges.plot(ax=ax, linewidth=0.8, color='gray')
gdf.plot(ax=ax, color='red', markersize=10)
plt.title("Street Map of Paris with Cartoradio Sites")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# 2 - Preparing sites and antennas databases with selection and filtering ================================================================

# Opening "antenne" file from pc directory 
df_antenne = pd.read_csv(
    '~/Documents/University/TelecomSudParis/PIR/cartoradio_ParisArea_06-11-25/Antennes_Emetteurs_Bandes_Cartoradio.csv', 
    delimiter=';', encoding='latin1')

# Selecting specific columns
df_sites = df_sites[['Numéro du support', 'Longitude', 'Latitude']]
df_sites = df_sites.rename(columns={'Numéro du support' : 'Numéro de support'})

df_antenne = df_antenne[['Numéro de support', 'Date de mise en service', 'Exploitant', 'Hauteur / sol', 
                         'Type service', 'Système', 'Début', 'Fin', 'Unité']]

# Filtering antenna characteristics for training

# 2.1 - Limit date for implementation
# Eliminate anything that is not in the date format
df_antenne['Date de mise en service'] = pd.to_datetime(df_antenne['Date de mise en service'], format="%d/%m/%Y", errors='coerce')
df_antenne = df_antenne.dropna(subset=['Date de mise en service'])
# Transform the date string into interpretable date and set limit
lim_date = pd.to_datetime("01/07/2023", format='%d/%m/%Y')
df_antenne = df_antenne[df_antenne['Date de mise en service'] < lim_date]

# 2.2 - Names of the carriers
carriers = ['ORANGE', 'BOUYGUES', 'SFR', 'FREE MOBILE']
df_antenne = df_antenne[df_antenne['Exploitant'].isin(carriers)]

# 2.3 - Height with respect to ground level
df_antenne = df_antenne[df_antenne['Hauteur / sol'] >= 0]

# 2.4 - Type of service
df_antenne = df_antenne[df_antenne['Type service'].isin(['TEL MOBILE'])]

# 2.5 - Selecting only downlink frequency bands
# Reference is Wikipedia "Fréquences de téléphonie mobile en France"
# Selection is done based on "Début" frequency
dl_orange = [763.0, 811.0, 934.9, 1805.0, 2154.9, 2635.0, 3710.0] # 1900 MHz band is not available (5G) on table
dl_bouygues = [773.0, 791.0, 925.1, 1860.0, 1900.1, 2125.3, 2655.0, 3570.0]
dl_sfr = [758.0, 801.0, 951.2, 1825.0, 1915.1, 2110.5, 2620.0, 3490.0]
dl_free = [778.0, 944.9, 1845.0, 2140.1, 2670.0, 3640.0] # 900 MHz band (944.9) doesn't match Wikipedia (943.5)

df_antenne = df_antenne[((df_antenne['Exploitant'] == 'ORANGE') & df_antenne['Début'].isin(dl_orange)) |
                        ((df_antenne['Exploitant'] == 'BOUYGUES') & df_antenne['Début'].isin(dl_bouygues)) |
                        ((df_antenne['Exploitant'] == 'SFR') & df_antenne['Début'].isin(dl_sfr)) |
                        ((df_antenne['Exploitant'] == 'FREE MOBILE') & df_antenne['Début'].isin(dl_free))]

# 2.6 - Merging sites and antennas databases
df_sit_ant = df_antenne.merge(df_sites, on='Numéro de support', how="left")

# Prints for checking
print(df_sites.head())
print(df_antenne.head())
print(df_sit_ant.head())

# 3 - Data exploitation from 5G data datasets =============================================================================================

# 3.1 - Creating the dataset by combining the different file groups (ftp, video, voice, voip)

# Selecting info to use in each 5G table
info_5G = ['Packet Tech', 'Band', 'Sidelink Band', 'Energy mJ', 'Sidelink Energy mJ',
                    'Total Energy mJ', 'Data Mbit', 'Sidelink Data Mbit', 'Total Data Mbit', 
                    'OperatorName', 'RSSI (pcell)', 'RSSI (scell)']

# Opening "data_5G" files from pc directory, from each group
files_5G = {
    'ftp' : '~/Documents/University/TelecomSudParis/PIR/data_5G/FTP/*5G*.csv',
    'video' : '~/Documents/University/TelecomSudParis/PIR/data_5G/Video/*5G*.csv',
    'voice' : '~/Documents/University/TelecomSudParis/PIR/data_5G/Voice/*5G*.csv',
    'voip' : '~/Documents/University/TelecomSudParis/PIR/data_5G/VOIP/*5G*.csv'
}

# Building the datasets with the corresponding files, and the specific columns for each
datasets_5G = {}
for key, pattern in files_5G.items():
    pattern = os.path.expanduser(pattern) # Use the os to acess a compatible file path
    files = sorted(glob.glob(pattern)) # The glob just gets the files, but they need to be sorted alphabetically
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        common_cols = [c for c in info_5G if c in df.columns]
        dfs.append(df[common_cols])
    datasets_5G[key] = dfs

# Check the first FTP dataset, for example
print(datasets_5G['ftp'][0].head())
