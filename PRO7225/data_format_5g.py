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
import re
from scipy import spatial

# 1 - Opening map files, display and plot ===============================================================================================

# Opening "Sites_Cartoradio" file from pc directory 
df_sites = pd.read_csv('~/Documents/University/TelecomSudParis/PIR/cartoradio_ParisArea_06-11-25/Sites_Cartoradio.csv',
                 delimiter=';', encoding='latin1')

# Defining delimitating latitudes and longitudes of the sites area
lat_max_sites = df_sites['Latitude'].max()
lat_min_sites = df_sites['Latitude'].min()
long_max_sites = df_sites['Longitude'].max()
long_min_sites = df_sites['Longitude'].min()

# Converting to library frame
gdf = gpd.GeoDataFrame(
    df_sites, geometry=gpd.points_from_xy(df_sites.Longitude, df_sites.Latitude), crs="EPSG:4326"
)

# Downloading street network and converting its graph
G = ox.graph_from_place('Paris, France', network_type='drive')
edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

# # Plot
# fig, ax = plt.subplots(figsize=(10, 6))
# edges.plot(ax=ax, linewidth=0.7, color='gray')
# gdf.plot(ax=ax, color='red', markersize=8)
# plt.title("Street Map of Paris with Cartoradio Sites")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.show()

# 2 - Preparing sites and antennas databases with selection and filtering ================================================================

# Opening "antenne" file from pc directory 
df_antenne = pd.read_csv(
    '~/Documents/University/TelecomSudParis/PIR/cartoradio_ParisArea_06-11-25/Antennes_Emetteurs_Bandes_Cartoradio.csv', 
    delimiter=';', encoding='latin1')

# Selecting specific columns
df_sites = df_sites[['Numéro du support', 'Longitude', 'Latitude']]
df_sites = df_sites.rename(columns={'Numéro du support' : 'Numéro de support'}) # Rename because of the difference in "du" vs "de"

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
# 1900 MHz frequency will be ignored (reserved to professional usage)
dl_orange = [763.0, 811.0, 934.9, 1805.0, 2154.9, 2635.0, 3710.0]
dl_bouygues = [773.0, 791.0, 925.1, 1860.0, 2125.3, 2655.0, 3570.0]
dl_sfr = [758.0, 801.0, 951.2, 1825.0, 2110.5, 2620.0, 3490.0]
dl_free = [778.0, 944.9, 1845.0, 2140.1, 2670.0, 3640.0] # 900 MHz band (944.9) doesn't match Wikipedia (943.5)

df_antenne = df_antenne[((df_antenne['Exploitant'] == 'ORANGE') & df_antenne['Début'].isin(dl_orange)) |
                        ((df_antenne['Exploitant'] == 'BOUYGUES') & df_antenne['Début'].isin(dl_bouygues)) |
                        ((df_antenne['Exploitant'] == 'SFR') & df_antenne['Début'].isin(dl_sfr)) |
                        ((df_antenne['Exploitant'] == 'FREE MOBILE') & df_antenne['Début'].isin(dl_free))]

# 2.6 - Merging sites and antennas databases
df_sit_ant = df_antenne.merge(df_sites, on='Numéro de support', how="left")

# Prints for checking
print(df_sites.head())
print("\n")
print(df_antenne.head())
print("\n")
print(df_sit_ant.head())
print("\n")

# 3 - Preparing table with information from 5G data datasets ===============================================================================

# 3.1 - Creating dataset by combining the different file types (ftp, video, voice, voip)

# Opening "data_5G" files from pc directory, from each group
files_5G = {
    'ftp' : '~/Documents/University/TelecomSudParis/PIR/data_5G/FTP/*5G*.csv',
    'video' : '~/Documents/University/TelecomSudParis/PIR/data_5G/Video/*5G*.csv',
    'voice' : '~/Documents/University/TelecomSudParis/PIR/data_5G/Voice/*5G*.csv',
    'voip' : '~/Documents/University/TelecomSudParis/PIR/data_5G/VOIP/*5G*.csv'
}

# Selecting info to use in each 5G table
info_5G = ['Duration s', 'Packet Tech', 'Band', 'Sidelink Band', 'Energy mJ', 'Sidelink Energy mJ',
                    'Total Energy mJ', 'OperatorName', 'RSRP (pcell).1', 'RSRP (scell).1']

# Building the datasets with the corresponding files, and the specific columns for each
datasets_5G = {}
for key, pattern in files_5G.items():
    pattern = os.path.expanduser(pattern) # Use the os to acess a compatible file path
    files = sorted(glob.glob(pattern)) # The glob just gets the files, but they need to be sorted alphabetically
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        common_cols = [col for col in info_5G if col in df.columns]
        df = df[common_cols]
        df.insert(0, "File type", key) # Add column with file type (ftp, ...) in the beginning
        # Extracting filename charactersitics based on "NoXXX_pY_5G.csv" name
        filename = os.path.basename(f)
        match = re.search(r'No(\d+)_p(\d+)_5G.csv', filename)
        if match:
            no_val = match.group(1) # = X
            p_val = match.group(2) # = Y
            file_id = f'{key}_{no_val}_{p_val}'
        else:
            file_id = f'{key}_{filename}'
        df.insert(1, 'File ID', file_id) # Add column with ID number to allow for ordering later
        dfs.append(df)
    datasets_5G[key] = dfs

# Check the first FTP dataset, for example
print(datasets_5G['ftp'][0].head())
print("\n")

# 3.2 - Merging datasets: each table from a single measurement becomes a row (or multiple rows) of larger dataset

# Criteria (to preserve file type order and merge according to distinguished information)
order_criteria = ['ftp', 'voice', 'video', 'voip']
merged_criteria = ['File ID', 'Packet Tech', 'Band', 'Sidelink Band']

# Concatenating the complete table from the filtered datasets, without merging
dfs_table = []
for key in order_criteria:
    for df in datasets_5G[key]:
        dfs_table.append(df)
complete_table = pd.concat(dfs_table)

# Checking if all numbers are represented as numbers and not as strings (in particular for RSRP data)
rsrp_cols = ['RSRP (pcell).1', 'RSRP (scell).1']
for col in rsrp_cols:
    complete_table[col] = pd.to_numeric(complete_table[col], errors='coerce')

# Converting every RSRP value to linear before summing, by creating a new auxiliary column
for col in rsrp_cols:
    if col in complete_table.columns:
        complete_table[f'{col}_linear'] = 10**(complete_table[col] / 10)

# Differentiating numbers from strings (auxiliary)
num_column = [col for col in complete_table.select_dtypes(include='number').columns
              if col not in merged_criteria]
str_column = [col for col in complete_table.select_dtypes(exclude='number').columns
              if col not in merged_criteria]

# Merging
merged_5g = {}
for col in num_column:
    merged_5g[col] = 'sum' # For columns with numbers, provide their sum
for col in str_column:
    merged_5g[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else None # For columns with strings, select the most frequent

# 3.3 - Building the final table to use in training
# A big table with all the merged information is built first, and then final adjustments are made

# Bulding
final_table = complete_table.groupby(merged_criteria).agg(merged_5g).reset_index()

# Converting every RSRP value back to dB after sum
for col in rsrp_cols:
    linear_col = f'{col}_linear'
    if linear_col in final_table.columns:
        final_table[col] = 10 * np.log10(final_table[linear_col])
        final_table = final_table.drop(columns=[linear_col]) # Removing the auxiliary linear column

# Filtering rows
final_table[final_table.eq(0)] = np.nan # Turning all zeros into empty information 
final_table = final_table.dropna() # Excluding rows with empty information

# Ordering according to file type
final_table[['File type', 'No_val', 'p_val']] = final_table['File ID'].str.extract(r'([a-z]+)_(\d+)_(\d+)$')
final_table['No_val'] = final_table['No_val'].astype('Int64') # Auxiliary column to sort in terms of No_val
final_table['p_val'] = final_table['p_val'].astype('Int64') # Auxiliary column to sort in terms of p_val

# Final table (in order)
final_table = final_table.sort_values(['File type', 'No_val', 'p_val', 'Packet Tech', 'Band', 'Sidelink Band'])
final_table = final_table.drop(columns=['No_val', 'p_val']) # Remove auxiliary columns 

# Print final table for check
print(final_table.head(12)) # Arbitrary number of rows
print("\n")

# 4 - Combining measurement description to Cartoradio map ================================================================================

# Opening "measurements_notes.csv" to get measurement location info
df_notes = pd.read_csv('~/Documents/University/TelecomSudParis/PIR/data_5G/Measurements/measurements_notes.csv',
                 delimiter=',', encoding='latin1')

# 4.1 - Formatting columns and rows

# Filtering columns of interest 
info_notes = ['Latitude', 'Longitude', 'Input Filename']
df_notes = df_notes[info_notes]

# Filtering rows of interest
df_notes = df_notes[(~df_notes['Input Filename'].str.contains('_4G.csv')) & # Exclude all 4g files, keeping only 5g
                    (df_notes['Latitude'].between(lat_min_sites, lat_max_sites)) & # Limit measurements to the sites' area
                    (df_notes['Longitude'].between(long_min_sites, long_max_sites))]

# 4.2 - Adding latitude and longitude columns to the final table by comparing filename to file ID

# Function to create matching ID
def extract_id_from_filename(filename):
    match = re.search(r'No(\d+)_p(\d+)_5G.csv', filename)
    if match:
        return f'{match.group(1)}_{match.group(2)}'
    return None

# Creating matching ID in df_notes
df_notes['Match ID'] = df_notes['Input Filename'].apply(extract_id_from_filename)
# Creating mathcin ID in final_table
final_table['Match ID'] = final_table['File ID'].str.extract(r'[a-z]+_(\d+_\d+)$')[0]

# Merging
final_table = final_table.merge(df_notes[['Match ID', 'Latitude', 'Longitude']], on='Match ID', how='left')

# Eliminating rows where there are NaN and auxiliary columns
# Here, the NaN means there was no correspondence between the two tables, which serves to eliminate from final_table
# all locations that are outside of the sites' area.
final_table = final_table.dropna()
final_table = final_table.drop(columns=['Match ID', 'File type'])

# Check
print(df_notes.head(20))
print("\n")
print(final_table.head(20))
print('\n')

# 4.3 - Plot of the Cartoradio map containing the measurement locations

# Converting to library frame
gdf_final = gpd.GeoDataFrame(
    final_table, geometry=gpd.points_from_xy(final_table.Longitude, final_table.Latitude), crs="EPSG:4326"
)

# # Plot
# fig, ax = plt.subplots(figsize=(10, 6))
# # Contour
# edges.plot(ax=ax, linewidth=0.7, color='gray', label='Streets')
# # Points
# gdf.plot(ax=ax, color='red', markersize=8, label='Cartoradio Sites')
# gdf_final.plot(ax=ax, color='blue', markersize=10, label='Measurement Points', marker='*')
# # Configuration
# plt.title("Street Map of Paris with Cartoradio Sites and Measurement Points")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.legend()
# plt.show()

# 4.4 - Calculating the distance from each measurement point to the nearest antenna
# Since
# a) the height of the antennas is in the order to 10^1 m, while the height of the measurement devices is 10^0 m and
# b) it's assumed all points are within than 10^1 km from one another
# the calculation of the distance will be a simple 2D euclidian

# Converting the distance between two points on Earth in degrees to kilometers
def haversine_distance(lat1, lon1, lat2, lon2):

    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371

    return c * r

# Calculating distance based on haversine_distance
def nearestDistance(df1, df2):

    # Building KD-tree from first dataframe to facilitate finding nearest neighbor search (NNS)
    tree = spatial.cKDTree(df1[['Latitude', 'Longitude']].values)
    # Performing NNS from dataframe 2
    distances, indices = tree.query(df2[['Latitude', 'Longitude']].values)

    # Adding columns with the indices, latitudes, and longitudes of nearest points
    result = df2.copy()
    result['Nearest BS Index'] = df1.index[indices]
    result['Nearest Latitude'] = df1.iloc[indices]['Latitude'].values
    result['Nearest Longitude'] = df1.iloc[indices]['Longitude'].values

    # Adding column with the corresponding distance to nearest (latitude, longitude)
    result['Distance [km]'] = haversine_distance(
        result['Latitude'].values,
        result['Longitude'].values,
        result['Nearest Latitude'].values,
        result['Nearest Longitude'].values
    )

    # Drop auxiliary columns
    # This can be done since the association of points will be done through coordinates
    result = result.drop(columns=['Nearest BS Index'])

    return result

# Building the dataframe 
min_distance_map = nearestDistance(df_sit_ant, final_table)
print(min_distance_map.head())

# Converting to library frame
gdf_minDistance = gpd.GeoDataFrame(
    min_distance_map, geometry=gpd.points_from_xy(min_distance_map.Longitude, min_distance_map.Latitude), crs="EPSG:4326"
)

# Plot with lines between measurement points and nearest BSs
fig, ax = plt.subplots(figsize=(10, 6))
# Contour
edges.plot(ax=ax, linewidth=0.7, color='gray', label='Streets')
# Points
gdf.plot(ax=ax, color='red', markersize=8, label='Cartoradio Sites')
gdf_final.plot(ax=ax, color='blue', markersize=10, label='Measurement Points', marker='*')
# Drawing lines between corresponding points using coordinates
for idx, row in gdf_minDistance.iterrows():
    # Getting coordinates from columns of nearest latitude and longitude
    nearest_lat = row['Nearest Latitude']  
    nearest_lon = row['Nearest Longitude']  
    distance = row['Distance [km]'] 
    # Getting coordinates of both points
    point1 = row.geometry  # Measurement point
    point2_x = nearest_lon
    point2_y = nearest_lat
    # Drawing line
    ax.plot([point1.x, point2_x], [point1.y, point2_y], 
            color='green', linestyle='--', linewidth=0.8, alpha=0.6)
    # # Adding distance label at midpoint
    # mid_x = (point1.x + point2_x) / 2
    # mid_y = (point1.y + point2_y) / 2
    # ax.text(mid_x, mid_y, f'{distance:.0f}m',  # Adjust format as needed
    #         fontsize=8, ha='center', va='bottom',
    #         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
# Configuration
plt.title("Street Map of Paris with Cartoradio Sites and Measurement Points")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()    
