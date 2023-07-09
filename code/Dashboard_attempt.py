from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd


import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

#import cdstoolbox as ct
#import chart_studio.plotly as py

import chart_studio.plotly as py
import dash_bootstrap_components as dbc




##### Figure #####


df = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/monthly_data.csv' )

#df2 = pd.read_fwf(
#    'C:/Users/bengo/OneDrive - Nexus365/ffs.txt',
#    index_col=0,
#    usecols=(0, 1),
#    names=['year', 'anomaly'],
#    header=None,
#)


df = df.dropna()

df = df[['date', 'ch4']]
df = df.groupby(['date']).mean()

#df[['ch4']] = df[['ch4']]*100000

df = df.assign(row_number=range(len(df)))




FIRST = 0
LAST = 226  # inclusive


# Reference period for the center of the color scale

FIRST_REFERENCE = 0
LAST_REFERENCE = 226
LIM = 0.01 # degrees

df = df.set_index('row_number')

ch4 = df.loc[FIRST:LAST, 'ch4'].dropna()
reference = ch4.loc[FIRST_REFERENCE:LAST_REFERENCE].mean()


df = df[[ 'ch4']]



fig = go.Figure()

fig.update_layout(
    width=800,
    height=350,
    xaxis=dict(
        title='Monthly Methane Levels',
        range=[FIRST, LAST + 1],
        showgrid=False,
        zeroline=False,
        tickmode='array',
        showticklabels=False,
        tickvals=[]
    ),
    yaxis=dict(
        range=[0, 1],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        tickvals=[]
    )
)


x_values = np.arange(FIRST, LAST + 1)
y_values = np.zeros_like(x_values)
z_values = np.array(ch4)

fig.add_trace(
    go.Heatmap(
        x=x_values,
        y=y_values,
        z=z_values,
        colorscale='rdbu_r',
        showscale=False
    )
)

fig.add_trace(
    go.Heatmap(
        x=x_values,
        y=y_values,
        z=z_values,
        colorscale='rdbu_r',
        showscale=False,
        hovertemplate='Month: %{text}<br>Methane Levels: %{z}<extra></extra>',
        text=[(datetime(2003, 1, 1) + timedelta(days=30 * (x - 1))).strftime('%B %Y') for x in range(FIRST, LAST + 1)]
    )
)


####Figure 2 - caro's mapppp#####

# Read the CSV file
final_daily2 = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/combined_daily.csv')
simulated = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/simulated_data.csv')
# Round 'lon_m' and 'lat_m' columns
final_daily2['lon_m'] = final_daily2['lon_m'].round()
final_daily2['lat_m'] = final_daily2['lat_m'].round()
# Convert 'time' column to datetime
final_daily2['date'] = pd.to_datetime(final_daily2['time']).dt.date
# Group by 'date', 'lon_m', and 'lat_m', and calculate the mean of 'ch4'
mean_ch4 = final_daily2.groupby(['date','lon_m', 'lat_m'])['ch4'].mean().reset_index()
# Remove rows with missing values
mean_ch4 = mean_ch4.dropna()

#convert date to date column
mean_ch4['date'] = pd.to_datetime(mean_ch4['date'])

#order data by date
mean_ch4.sort_values(by='date', ascending=False, inplace=True)
#keep only most recent data for each lon and lat
mean_ch4.drop_duplicates(subset=['lon_m', 'lat_m'], keep='first', inplace=True)
#reset
mean_ch4.reset_index(drop=True, inplace=True)

import plotly.graph_objects as go
from scipy.spatial import cKDTree

import plotly.express as px
import plotly.graph_objects as go

# Create a Plotly choropleth map


import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import cdist
import numpy as np

hospitals = simulated[simulated['plant_or_hospital'] == "Hospital"]
hospitals = hospitals.rename(columns={'latitude': 'lat'})
hospitals = hospitals.rename(columns={'longitude': 'lon'})

energy_plants = simulated[simulated['plant_or_hospital'] == "Energy plant"]
energy_plants = energy_plants.rename(columns={'latitude': 'lat'})
energy_plants = energy_plants.rename(columns={'longitude': 'lon'})

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from scipy.spatial import cKDTree


df_sorted = mean_ch4.sort_values('ch4', ascending=False)

# Calculate the number of rows representing the top 10%
top_10_percent = int(len(df_sorted) * 0.1)

# Slice the DataFrame to keep only the top rows
top_10_df = df_sorted.head(top_10_percent)

n = len(top_10_df)

# Create a new column 'Ascending' with values from 1 to n
top_10_df['USA_Methane_Ranking'] = range(1, n+1)

top_10_df = top_10_df.rename(columns={'lat_m': 'lat'})
top_10_df = top_10_df.rename(columns={'lon_m': 'lon'})


from scipy.spatial.distance import cdist

# Assuming 'hospitals' and 'top_10_df' are your DataFrames
# Assuming 'hospitals' has columns 'Hospital Name', 'Latitude', and 'Longitude'
# Assuming 'top_10_df' has columns 'Row ID', 'Latitude', and 'Longitude'

# Create a helper function to find the nearest hospital for a given location
def find_nearest_hospital(row, hospitals):
    # Create a DataFrame with coordinates of the current row
    coords = pd.DataFrame({'lat': [row['lat']], 'lon': [row['lon']]})

    # Calculate the distances between the coordinates of the current row and all hospitals
    distances = cdist(coords.values, hospitals[['lat', 'lon']].values, metric='euclidean')

    # Find the index of the nearest hospital
    nearest_index = distances.argmin()

    # Return the name of the nearest hospital
    return hospitals.iloc[nearest_index]['company_name']

# Apply the helper function to each row in 'top_10_df'
top_10_df['Nearest_Hospital'] = top_10_df.apply(lambda row: find_nearest_hospital(row, hospitals), axis=1)



# Create a helper function to find the nearest hospital for a given location
def find_nearest_plant(row, energy_plants):
    # Create a DataFrame with coordinates of the current row
    coords = pd.DataFrame({'lat': [row['lat']], 'lon': [row['lon']]})

    # Calculate the distances between the coordinates of the current row and all hospitals
    distances = cdist(coords.values, energy_plants[['lat', 'lon']].values, metric='euclidean')


    # Find the index of the nearest hospital
    nearest_index = distances.argmin()

    # Return the name of the nearest hospital
    return energy_plants.iloc[nearest_index]['company_name']

# Apply the helper function to each row in 'top_10_df'
top_10_df['Nearest_Energy_plant'] = top_10_df.apply(lambda row: find_nearest_plant(row, energy_plants), axis=1)





min_value = top_10_df['ch4'].min()
max_value = top_10_df['ch4'].max()
scaled_values = ((top_10_df['ch4'] - min_value) / (max_value - min_value)) ** 1.2 * 15

customdata = top_10_df[['USA_Methane_Ranking', 'Nearest_Hospital', 'Nearest_Energy_plant', 'ch4']].values.tolist()
customdata = np.stack((top_10_df['USA_Methane_Ranking'], top_10_df['Nearest_Hospital'], top_10_df['Nearest_Energy_plant'], top_10_df['ch4']), axis=-1)


# Create the scatter plot with scaled point sizes
fig2 = px.scatter_mapbox(top_10_df, lat="lat", lon="lon", hover_name="ch4",
                         size=scaled_values,
                         color_continuous_scale='viridis',
                         mapbox_style="carto-positron",
                         zoom=2,
                         center={'lat': 39.8283, 'lon': -98.5795},
                         opacity=0.8, size_max=30,    
                         custom_data=['USA_Methane_Ranking', 'Nearest_Hospital', 'Nearest_Energy_plant', 'ch4'])


fig2.update_traces(hovertemplate='Methane ranking: %{customdata[0]}<br>Nearest Hospital: %{customdata[1]}<br>Nearest Energy Plant: %{customdata[2]}<br>Methane particles (Mole fraction): %{customdata[3]}')





####Figure 3, Dunja Map####

methane_df = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/Methane.txt', delimiter='\t')
mh_df = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/MentalHealth.txt', delimiter='\t')
respiratory_df = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/Respiratory.txt', delimiter='\t')

# Add identifiers to each df
methane_df['ID'] = 'methane'
mh_df['ID'] = 'mh'
respiratory_df['ID'] = 'respiratory'

# Merge dfs to one 
health_df = pd.concat([methane_df, mh_df, respiratory_df], axis=0, ignore_index=True)
#Get USA County Coordinates
import geopandas as gpd
import matplotlib.pyplot as plt
import geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
#Use geopy library to generate map!
geolocator = Nominatim(user_agent="my_geocoder")
def get_coordinates(county):
    try:
        location = geolocator.geocode(county)
        return location.latitude, location.longitude
    except GeocoderTimedOut:
        return get_coordinates(county)
county_names = health_df["County"].unique()
county_names = county_names.tolist()

coordinates = []
#for county in county_names:
#    try:
#        lat, lon = get_coordinates(county)
#        coordinates.append((county, lat, lon))
#    except AttributeError:
#        print(f"Invalid county name: {county}. Skipping...")
        
        
# Invalid county name: Hoonah-Angoon Census Area, AK. Skipping...
# Invalid county name: Petersburg Borough/Census Area, AK. Skipping...
# Invalid county name: De Kalb County, IN. Skipping...
# Invalid county name: Ste. Genevieve County, MO. Skipping...
# Invalid county name: Bethel Census Area, AK. Skipping...
# Invalid county name: Prince of Wales-Hyder Census Area, AK. Skipping...
# Invalid county name: Southeast Fairbanks Census Area, AK. Skipping...
# Invalid county name: Valdez-Cordova Census Area, AK. Skipping...
# Invalid county name: Yukon-Koyukuk Census Area, AK. Skipping...


df_coordinates = pd.DataFrame(coordinates, columns=['County', 'Latitude', 'Longitude'])
#Manually do the ones that failed/where geopy failed me
# manually get the rest of the coordinates because geopy failed me 
additional = [['Buena Vista city, VA', -79.3524, 37.7348],
['Charlottesville city, VA', -78.4767, 38.0293],
['Colonial Heights city, VA', -77.4086, 37.2440],
['Covington city, VA', -79.9916, 37.7840],
['Emporia city, VA', -77.5425, 36.6966],
['Falls Church city, VA', -77.1711, 38.8823],
['Franklin city, VA', -76.9366, 36.6844],
['Galax city, VA', -80.9170, 36.6658],
['Harrisonburg city, VA', -78.8689, 38.4496],
['Lexington city, VA', -79.4428, 37.7840],
['Manassas Park city, VA', -77.4525, 38.7726],
['Norton city, VA', -82.6281, 36.9323],
['Poquoson city, VA', -76.3488, 37.1224],
['Radford city, VA', -80.5764, 37.1206],
['Staunton city, VA', -79.0717, 38.1496],
['Waynesboro city, VA', -78.8867, 38.0685],
['Williamsburg city, VA', -76.7085, 37.2707],
['Adams County, WA', -118.4731, 46.9906],
['Asotin County, WA', -117.1965, 46.1902],
['Ferry County, WA', -118.5463, 48.4690],
['Klickitat County, WA', -120.7752, 45.8738],
['Lincoln County, WA', -118.4148, 47.5796],
['Pacific County, WA', -123.8380, 46.5267],
['Pend Oreille County, WA', -117.2664, 48.5252],
['San Juan County, WA', -123.0118, 48.6006],
['Skamania County, WA', -121.9169, 46.0240],
['Whitman County, WA', -117.5244, 46.8930],
['Doddridge County, WV', -80.7072, 39.2670],
['Pendleton County, WV', -79.3634, 38.7058],
['Pleasants County, WV', -81.1703, 39.3581],
['Pocahontas County, WV', -80.0388, 38.3719],
['Ritchie County, WV', -81.0976, 39.2111],
['Roane County, WV', -81.3972, 38.7490],
['Tucker County, WV', -79.6430, 39.1128],
['Tyler County, WV', -80.8982, 39.4393],
['Webster County, WV', -80.4366, 38.4812],
['Wirt County, WV', -81.4142, 39.0423],
['Bayfield County, WI', -91.1690, 46.484],
 ['Buffalo County, WI', -91.7501, 44.3638],
['Burnett County, WI', -92.3632, 45.8685],
['Clark County, WI', -90.5943, 44.7342],
['Crawford County, WI', -90.9500, 43.2304],
['Door County, WI', -87.0524, 45.0428],
['Dunn County, WI', -91.8952, 44.9517],
['Florence County, WI', -88.4004, 45.8415],
['Forest County, WI', -88.7820, 45.6526],
['Grant County, WI', -90.7012, 42.8624],
['Green County, WI', -89.5996, 42.6793],
['Green Lake County, WI', -89.0198, 43.8108],
['Iowa County, WI', -90.1384, 43.0013],
['Iron County, WI', -90.4620, 46.3145],
['Kewaunee County, WI', -87.6137, 44.4735],
['Lafayette County, WI', -90.1534, 42.6659],
['Langlade County, WI', -89.1044, 45.2407],
['Lincoln County, WI', -89.7229, 45.3565],
['Marquette County, WI', -89.4056, 43.8116],
['Menominee County, WI', -88.7634, 44.9084],
['Oneida County, WI', -89.4952, 45.7074],
['Pepin County, WI', -92.0010, 44.6121],
['Pierce County, WI', -92.4102, 44.7243],
['Polk County, WI', -92.3835, 45.4694],
['Price County, WI', -90.3893, 45.7003],
['Richland County, WI', -90.4471, 43.3813],
['Rusk County, WI', -91.1122, 45.4689],
['Taylor County, WI', -90.5306, 45.2272],
['Trempealeau County, WI', -91.3476, 44.3070],
['Vernon County, WI', -90.8498, 43.5884],
['Vilas County, WI', -89.3088, 46.0519],
['Washburn County, WI', -91.8117, 45.8526],
['Waushara County, WI', -89.2524, 44.1129],
['Albany County, WY', -105.7529, 41.6511],
['Big Horn County, WY', -107.9694, 44.5232],
['Carbon County, WY', -107.3170, 41.7036],
['Converse County, WY', -105.5342, 42.9949],
['Goshen County, WY', -104.3835, 42.0822],
['Johnson County, WY', -106.5970, 44.0494],
 ['Lincoln County, WY', -110.5667, 42.8333],
['Park County, WY', -109.5967, 44.5236],
['Platte County, WY', -104.7000, 42.1000],
['Sheridan County, WY', -107.0336, 44.8667],
['Sublette County, WY', -109.9154, 42.8500],
['Teton County, WY', -110.5885, 43.8537],
['Uinta County, WY', -110.9649, 41.2833],
['Washakie County, WY', -107.6780, 43.9167],
['Weston County, WY', -104.5321, 43.9167],
['Crowley County, CO', -103.7833, 38.3333],
['Clinch County, GA', -82.8790, 30.9632],
['Echols County, GA', -83.0922, 30.6976],
['Lincoln County, GA', -82.4346, 33.7914],
['Pulaski County, GA', -83.4994, 32.2304],
['Stewart County, GA', -84.8429, 32.0886],
['Talbot County, GA', -84.5333, 32.7167],
['Taliaferro County, GA', -82.8982, 33.5471],
['Butte County, ID', -113.1761, 43.6283],
['Pope County, IL', -88.5167, 37.4000],
['Gove County, KS', -100.4722, 38.8833],
['Haskell County, KS', -100.8833, 37.5500],
['Wichita County, KS', -101.3667, 38.4500],
['Robertson County, KY', -84.1461, 38.5837],
['Tensas Parish, LA', -91.3247, 32.0112],
['Sharkey County, MS', -90.8275, 32.9264],
['Tunica County, MS', -90.3749, 34.6498],
['Knox County, MO', -92.1513, 40.1392],
['Mercer County, MO', -93.5644, 40.4211],
['Wheatland County, MT', -109.9603, 46.4609],
['Garden County, NE', -102.3406, 41.5962],
['Hitchcock County, NE', -101.0000, 40.1833],
['Perkins County, NE', -101.6167, 40.8500],
['Valley County, NE', -99.8500, 41.4000],
['Union County, NM', -103.4319, 36.5181],
['Sargent County, ND', -97.6358, 46.1071],
['Sioux County, ND', -101.0722, 46.1000],
['Cimarron County, OK', -102.6000, 36.7800],
['Roger Mills County, OK', -99.6608, 35.6583],
['Aurora County, SD', -98.5256, 43.7128],
['Hyde County, SD', -99.4028, 44.5413],
['Mellette County, SD', -100.7339, 43.5333],
['Stanley County, SD', -100.7761, 44.4089],
['Ziebach County, SD', -101.9072, 45.0167],
['Bailey County, TX', -102.8319, 34.0511],
['Baylor County, TX', -99.2400, 33.5800],
['Carson County, TX', -101.3742, 35.4000],
['Castro County, TX', -102.2500, 34.5167],
['Crockett County, TX', -101.3981, 30.8248],
['Crosby County, TX', -101.3817, 33.6258],
['Foard County, TX', -99.7794, 33.9836],
['Hemphill County, TX', -100.2583, 35.8167],
['Kimble County, TX', -99.7436, 30.4770],
['Knox County, TX', -99.7389, 33.6044],
['Lipscomb County, TX', -100.2661, 36.2833],
['Lynn County, TX', -101.7008, 33.1636],
['Reagan County, TX', -101.4903, 31.3204],
['Upton County, TX', -102.4333, 31.4333],
['Piute County, UT', -112.1115, 38.2667],
['Columbia County, WA', -117.9278, 46.3025],
['Wahkiakum County, WA', -123.4250, 46.2947],
['Gilmer County, WV', -80.8675, 38.9053],
['Niobrara County, WY', -104.4736, 43.1667],
['Hoonah-Angoon Census Area, AK', -135.3500, 58.1000],
['Petersburg Borough/Census Area, AK', -132.8339, 56.8122],
['De Kalb County, IN', -85.0067, 41.4073],
['Ste. Genevieve County, MO', -90.1667, 37.9167],
['Bethel Census Area, AK', -161.5142, 60.6890],
['Prince of Wales-Hyder Census Area, AK', -132.8961, 55.6247],
['Southeast Fairbanks Census Area, AK', -142.0675, 63.7361],
['Valdez-Cordova Census Area, AK', -144.7630, 61.4272],
['Yukon-Koyukuk Census Area, AK', -153.2211, 65.0319]]
# swap longitude and latitude to match rest of coordinates
for item in additional:
    item[-2], item[-1] = item[-1], item[-2]
df_additional = pd.DataFrame(additional, columns=['County', 'Latitude', 'Longitude'])
final_coordinates = pd.concat([df_coordinates, df_additional], ignore_index = True)
df = pd.merge(health_df, final_coordinates, on='County')

# filter datafame by ignoring deaths <= 9 or NaN
# also drop rows where County is nan
df = df.dropna(subset=['County'])
df = df.dropna(subset=['Deaths'])
df = df[df['Deaths'] >= 10]
# Renaming content and columns
df['ID'] = df['ID'].replace({'mh': 'Mental Health-related',
                             'respiratory': 'Respiratory Disease',
                             'methane': 'Methane/Gas-related'})

df = df.rename(columns={'ID': 'Cause of Death'})
#Plotting!
#Plot #1: Deaths by USA County
#Variable dot size (size depends on death column such that more deaths = bigger dot)

# Assuming you have the data in a DataFrame called 'df'
fig3 = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", hover_name="County",
                        color="Cause of Death", size="Deaths",
                        color_continuous_scale=[[0, "rgb(68, 1, 84)"],
                                                [0.25, "rgb(72, 35, 116)"],
                                                [0.5, "rgb(64, 67, 135)"],
                                                [0.75, "rgb(52, 94, 141)"],
                                                [1, "rgb(41, 121, 142)"]],
                        mapbox_style="carto-positron",
                        zoom=3, center={"lat": 37.0902, "lon": -95.7129},
                        opacity=0.8, size_max=30)

fig3.update_layout(title="Number of Deaths by Cause of Death in USA Counties",
                  height=600, width=900)









####fig 4 - deaths by month and county####



import pandas as pd
import plotly.express as px
# Load in the relevant dfs
methane_df = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/Month_Methane.txt', delimiter='\t')
mh_df_1 = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/2018_2019_Month_MentalHealth.txt', delimiter='\t')
mh_df_2 = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/2020_2021_Month_MentalHealth.txt', delimiter='\t')
respiratory_df_1 = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/2018_Month_Respiratory.txt', delimiter='\t')
respiratory_df_2 = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/2019_2020_Month_Respiratory.txt', delimiter='\t')
respiratory_df_3 = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/2021_Month_Respiratory.txt', delimiter='\t')

# Add identifiers to each df
methane_df['ID'] = 'methane'
mh_df_1['ID'] = 'mh'
mh_df_2['ID'] = 'mh'
respiratory_df_1['ID'] = 'respiratory'
respiratory_df_2['ID'] = 'respiratory'
respiratory_df_3['ID'] = 'respiratory'

respiratory_df_1['Year'] = 2018

# Merge dfs to one 
mh_df = pd.concat([mh_df_1, mh_df_2], axis=0, ignore_index=True)
respiratory_df = pd.concat([respiratory_df_1, respiratory_df_2, respiratory_df_3], axis=0, ignore_index=True)
month_health_df = pd.concat([methane_df, mh_df, respiratory_df], axis=0, ignore_index=True)

# Convert Month column to proper format
month_health_df['Month'] = month_health_df['Month'].str.split(',').str[0].str.strip()
# drop montha and county that are nan and deaths that are < 10
month_health_df = month_health_df.dropna(subset=['Month', 'County']).query('Deaths >= 10')
# Convert the 'Month' column to a categorical data type to ensure correct ordering
month_health_df['Month'] = pd.Categorical(month_health_df['Month'], categories=['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.'], ordered=True)

fig4 = px.scatter(month_health_df, x='Month', y='Deaths', color='County', symbol='County', 
                 color_continuous_scale='viridis', title='Number of Deaths by County by Month',
                 category_orders={'Month': ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'Jun.', 'Jul.', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.']})

fig4.update_layout(xaxis_title='Month', yaxis_title='Deaths')

fig4.update_layout(
    annotations=[
        dict(
            x=0,
            y=1.1,
            xref="paper",
            yref="paper",
            text="Cause of death: Mental health-related, respiratory disease-related, or methane/gas-related",
            showarrow=False,
            font=dict(size=10),
            align="center"
        )
    ]
)




####methane over time#####

df_county = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/methane_monthly_county.csv')

df_county['date'] = pd.to_datetime(df_county['date'])

FipsDF = pd.read_csv('https://raw.githubusercontent.com/BenGoodair/Methane_Dashboard/main/fips2county.tsv', sep='\t', header='infer', dtype=str, encoding='latin-1')
StateAbbrDF = FipsDF[["StateAbbr", 'StateFIPS', "StateName"]].drop_duplicates()

df_county = pd.merge(df_county, StateAbbrDF.astype({'StateFIPS': 'int64'}), left_on='STATEFP', right_on='StateFIPS', how='left')
df_county['StateName'] = df_county['StateName'].astype('object')
df_county = df_county.dropna(subset=['StateName'])
df_county['date'] = pd.to_datetime(df_county['date'])


#fig5 = px.scatter(df_county, x='date', y='ch4', color='ch4', trendline='lowess')
#fig5.update_traces(marker=dict(size=5))
#fig5.update_layout(
#  xaxis_title='Year',
#  yaxis_title='Methane ppm',
#  title='Rising methane emissions in the United States (2003-2021)',
#  coloraxis_colorbar=dict(title='Methane ppb')
#)

#df_county['date'] = pd.to_datetime(df_county['date'])




####map slider#####

import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime

# Read the data
df_daily = pd.read_csv('/Users/carolinkroeger/Library/CloudStorage/OneDrive-Nexus365/Projekte/Wellcome Ideathon/methane_processing/final_csvs/methane_final_lonlat.csv')
states = gpd.read_file('/Users/carolinkroeger/Library/CloudStorage/OneDrive-Nexus365/Projekte/Wellcome Ideathon/methane_processing/data/gadm36_USA_shp/gadm36_USA_1.shp')

FipsDF = pd.read_csv('/Users/carolinkroeger/Library/CloudStorage/OneDrive-Nexus365/Projekte/Wellcome Ideathon/methane_processing/data/fips2county.tsv', sep='\t', header='infer', dtype=str, encoding='latin-1')
StateAbbrDF = FipsDF[["StateAbbr", 'StateFIPS', "StateName"]].drop_duplicates()

df_daily = pd.merge(df_daily, StateAbbrDF.astype({'StateFIPS': 'int64'}), left_on='STATEFP', right_on='StateFIPS', how='left')
df_daily['StateName'] = df_daily['StateName'].astype('object')
df_daily = df_daily.dropna(subset=['StateName'])

# Filter and sample the data
df_daily['date'] = pd.to_datetime(df_daily['date'])
df_daily['day_of_year'] = df_daily['date'].dt.dayofyear
df_daily['week_of_year'] = df_daily['date'].dt.isocalendar().week

df_filtered = df_daily[df_daily['date'].dt.year == 2019]

























####Dashboard####
#app = Dash(__name__)
import dash 
from dash import dash_table
import reverse_geocoder as rg


app = dash.Dash(external_stylesheets=[dbc.themes.LUX])


#server = app.server


tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}



SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Climate Coders Methane Dashboard", className="display-7"),
        html.Hr(),
        html.P(
            "Welcome to a dashboard detailing the health impacts of methane exposure in the USA.", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Methane Emmissions", href="/page-1", active="exact"),
                dbc.NavLink("Respiratory Health", href="/page-2", active="exact"),
                dbc.NavLink("Links To Resources", href="/page-3", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


def generate_top_locations_table():
    # Sort the dataframe by 'ch4' column in descending order and select the top 10 rows
    top_locations = top_10_df.sort_values('ch4', ascending=False).head(10)
    
    # Use reverse geocoding to obtain the US state based on latitude and longitude
    coordinates = list(zip(top_locations['lat'], top_locations['lon']))
    state_info = rg.search(coordinates)
    us_states = [info['admin1'] for info in state_info]
    
    # Add the 'US State' column to the dataframe
    top_locations['US State'] = us_states

    top_locations.columns = [col.replace('_', ' ') for col in top_locations.columns]
    
    
 

    # Rename 'ch4' column to 'Methane particles (Mole fraction)'
    top_locations.rename(columns={'ch4': 'Methane particles (Mole fraction)'}, inplace=True)
    
    
    # Create a Dash DataTable component
    table = dash_table.DataTable(
        id='top-locations-table',
        columns=[{'name': col, 'id': col} for col in top_locations.columns],
        data=top_locations.to_dict('records'),
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
        style_table={'overflowX': 'auto'},
    )
    
    return table


content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.Div([
            html.H2("Welcome to the Climate Coders Methane Dashboard", className="display-7"),
            html.Hr(),
            html.H4("Purpose of the Dashboard"),
            html.P("The Climate Coders Methane Dashboard aims to provide policymakers with valuable insights into methane levels and their impact on population health in the USA. By visualizing methane emissions, respiratory health data, and related information, this dashboard assists policymakers in making informed decisions to address the challenges posed by methane exposure."),
            html.H4("Key Features"),
            html.Ul([
                html.Li("Methane Map: Interactive map showcasing methane levels across different locations in the USA."),
                html.Li("Methane Graphs: Graphical representations and visualizations of methane data trends."),
                html.Li("Health Map: Map displaying mortalities from respiratory diseases and gas leaks."),
                html.Li("Health Graphs: Graphs presenting respiratory health data and trends."),
                html.Li("Links to Resources: Comprehensive list of resources, papers, and articles related to methane emissions and respiratory health."),
            ]),
            html.H4("How to Use"),
            html.P("Navigate through the tabs at the sidebar to access different sections of the dashboard. Each section provides specific information and visualizations related to methane levels and health impacts. Use the interactive components to explore the data and gain insights."),
            html.H4("For Policymakers"),
            html.P("We encourage policymakers to utilize this dashboard as a resource for evidence-based decision-making. By considering the data, visualizations, and resources provided here, policymakers can better understand the magnitude of methane emissions and the potential health risks associated with it. Additionally, we recommend referring to the 'Links to Resources' section for further in-depth research and reports."),
            html.Hr(),
            html.H4("Important Note"),
            html.P("This dashboard is for informational purposes only and should not be used as the sole basis for policymaking. It is crucial to consult domain experts, conduct further analysis, and consider additional factors when making policy decisions."),
            html.Hr(),
            html.H5("For more information, please visit the following pages:"),
            dbc.Nav(
                [
                    dbc.NavLink("Methane Emissions", href="/page-1", active="exact"),
                    dbc.NavLink("Respiratory Health", href="/page-2", active="exact"),
                    dbc.NavLink("Links to Resources", href="/page-3", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
            html.Hr(),
            html.P("Acknowledgements, data sources, and contact information."),
        ], style={"padding": "2rem"})
    elif pathname == "/page-1":
        return html.Div([
            dcc.Tabs(id="page-1-tabs", value='tab-1', children=[
                dcc.Tab(label='Methane leaks', value='tab-1', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Methane levels by state', value='tab-2', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Methane map slider', value='tab-3', style=tab_style, selected_style=tab_selected_style),
            ], style=tabs_styles),
            html.Div(id='page-1-tabs-content')
        ])
    elif pathname == "/page-2":
        return html.Div([
            dcc.Tabs(id="page-2-tabs", value='tab-4', children=[
                dcc.Tab(label='Health Map', value='tab-4', style=tab_style, selected_style=tab_selected_style),
                dcc.Tab(label='Health Graphs', value='tab-5', style=tab_style, selected_style=tab_selected_style),
            ], style=tabs_styles),
            html.Div(id='page-2-tabs-content')
        ])
    elif pathname == "/page-3":
        return html.P("List of resources and links to papers, articles, etc.")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognized..."),
        ],
        className="p-3 bg-light rounded-3",
    )


@app.callback(Output('page-1-tabs-content', 'children'), [Input('page-1-tabs', 'value')])
def render_page_1_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Map of Methane Levels, overlayed with energy plant and hospital locations'),
            dcc.Graph(id='map'),
            html.Hr(),
            html.H4('Top Locations by Methane Levels'),
            generate_top_locations_table()
         ])
    if tab == 'tab-2':
        return html.Div([
             html.H3('Methane data visualizations'),
             dcc.Dropdown(
                id='county-dropdown',
                options=[{'label': StateName, 'value': StateName} for StateName in sorted(df_county['StateName'].unique())],
                value=None,
                placeholder='Select a state',
                style={'width': '200px', 'margin-bottom': '20px'}
            ),
            dcc.Graph(
                id='scatter-plot'
            )
])
#       dcc.Graph(id='Methane Stripes', figure=fig)
        ])
#    elif tab == 'tab-3':
#        return html.Div([
#            html.H3('Methane data slider'),
#            dcc.Graph(id='map-graph'),
#            dcc.Slider(
#        id='date-slider',
#        min=df_daily['year'].min(),
#        max=df_daily['year'].max(),
#        value=df_filtered['year'].min(),
#        marks={str(year): str(year) for year in df_daily['year'].unique()},
#        step=None
#    )
#        ])


@app.callback(Output('page-2-tabs-content', 'children'), [Input('page-2-tabs', 'value')])
def render_page_2_content(tab):
    if tab == 'tab-4':
        return html.Div([
            html.H3('Map of Mortalities from respiratory diseases and gas leaks'),
            dcc.Graph(id='Health map', figure=fig3)

        ])
    elif tab == 'tab-5':
        return html.Div([
            html.H3('Health data visualizations'),
            dcc.Graph(id='Health visualisation', figure=fig4)

        ])




def find_nearest_point(lat, lon, df):
    point = np.array([[lat, lon]])
    distances = cdist(point, df[['lat', 'lon']])
    nearest_index = np.argmin(distances)
    nearest_point = df.iloc[nearest_index]
    return nearest_point


# Define the callback function to update the scatter plot based on the dropdown selection
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('county-dropdown', 'value')]
)
def update_scatter_plot(selected_county):
    if selected_county is None:
        filtered_df = df_county
    else:
        filtered_df = df_county[df_county['StateName'] == selected_county]

    fig5 = px.scatter(filtered_df, x='date', y='ch4', color='ch4', trendline='lowess', range_color=[df_county['ch4'].min(), df_county['ch4'].max()])
    fig5.update_traces(marker=dict(size=5))
    fig5.update_layout(
        xaxis_title='Year',
        yaxis_title='Methane ppm',
        title='Rising methane emissions in the United States (2003-2021)',
        coloraxis_colorbar=dict(title='Methane ppb')
    )
    
    return fig5

@app.callback(Output('map', 'figure'), [Input('map', 'clickData')])
def update_nearest_hospital(click_data):
    if click_data:
        point = click_data['points'][0]

        lat = point['lat']
        lon = point['lon']

        nearest_hospital = find_nearest_point(lat, lon, hospitals)
        nearest_energy_plant = find_nearest_point(lat, lon, energy_plants)

        # Create a new figure with the nearest hospital and energy plant annotations
        new_fig = go.Figure(fig2)

        new_fig.add_trace(
            go.Scattermapbox(
                lat=[lat, nearest_hospital['lat']],
                lon=[lon, nearest_hospital['lon']],
                mode='lines',
                line=dict(color='red', width=2),
                hoverinfo='none'
            )
        )
        new_fig.add_trace(
            go.Scattermapbox(
                lat=[lat, nearest_energy_plant['lat']],
                lon=[lon, nearest_energy_plant['lon']],
                mode='lines',
                line=dict(color='blue', width=2),
                hoverinfo='none'
            )
        )
        new_fig.add_trace(
            go.Scattermapbox(
                lat=[nearest_hospital['lat']],
                lon=[nearest_hospital['lon']],
                mode='markers',
                customdata=[
                    nearest_hospital['company_name'],
                    nearest_hospital['phone_number'],
                    nearest_hospital['number_of_employees'],
                    nearest_hospital['previous_leaks_n'],
                    nearest_hospital['fossil_fuel_type'],
                    nearest_hospital['number_of_beds']
                ],
                marker=dict(size=10, color='red'),
                hovertemplate='Company Name: %{customdata[0]}<br>Company telephone: %{customdata[1]}<br>Number of Employees: %{customdata[2]}<br>Number of Hospital Beds: %{customdata[5]}'
            )
        )

        new_fig.add_trace(
            go.Scattermapbox(
                lat=[nearest_energy_plant['lat']],
                lon=[nearest_energy_plant['lon']],
                mode='markers',
                customdata=[
                    nearest_energy_plant['company_name'],
                    nearest_energy_plant['phone_number'],
                    nearest_energy_plant['number_of_employees'],
                    nearest_energy_plant['previous_leaks_n'],
                    nearest_energy_plant['fossil_fuel_type']
                ],
                marker=dict(size=10, color='blue'),
                hovertemplate='Company Name: %{customdata[0]}<br>Company telephone: %{customdata[1]}<br>Number of Employees: %{customdata[2]}<br>Number of previous methane leaks: %{customdata[3]}<br>Fossil fuel type: %{customdata[4]}'
            )
        )
       # Set the zoom and view location from the previous figure
        new_fig.update_layout(mapbox=dict(center=dict(lat=fig2['layout']['mapbox']['center']['lat'],
                                                      lon=fig2['layout']['mapbox']['center']['lon']),
                                               zoom=fig2['layout']['mapbox']['zoom']))

        return new_fig

    return fig2


#@app.callback(
#    dash.dependencies.Output('map-graph', 'figure'),
#    [dash.dependencies.Input('date-slider', 'value')]
#)
#def update_map(selected_year):
#    filtered_data = df_daily[df_daily['year'] == selected_year]###
#
#    fig6 = px.scatter_mapbox(filtered_data, lat='latitude', lon='longitude',
#                            hover_data=['ch4'], color='ch4', color_continuous_scale='Viridis',
#                            size='ch4', size_max=10, zoom=3)
#    
#    fig6.update_layout(mapbox_style='open-street-map')
#    fig6.update_traces(marker=dict(opacity=0.8))
#    fig6.update_layout(legend=dict(x=0, y=1), margin=dict(l=0, r=0, t=0, b=0))
#    
#    return fig6



if __name__ == '__main__':
    app.run_server(host='localhost',port=8005)





























app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

# Define the layout of the app
app.layout = html.Div([
    dcc.Graph(id='map')
])

def find_nearest_point(lat, lon, df):
    point = np.array([[lat, lon]])
    distances = cdist(point, df[['lat', 'lon']])
    nearest_index = np.argmin(distances)
    nearest_point = df.iloc[nearest_index]
    return nearest_point


@app.callback(Output('map', 'figure'), [Input('map', 'clickData')])
def update_nearest_hospital(click_data):
    if click_data:
        point = click_data['points'][0]

        lat = point['lat']
        lon = point['lon']

        nearest_hospital = find_nearest_point(lat, lon, hospitals)
        nearest_energy_plant = find_nearest_point(lat, lon, energy_plants)

        # Create a new figure with the nearest hospital and energy plant annotations
        new_fig = go.Figure(fig2)

        new_fig.add_trace(
            go.Scattermapbox(
                lat=[lat, nearest_hospital['lat']],
                lon=[lon, nearest_hospital['lon']],
                mode='lines',
                line=dict(color='red', width=2),
                hoverinfo='none'
            )
        )
        new_fig.add_trace(
            go.Scattermapbox(
                lat=[lat, nearest_energy_plant['lat']],
                lon=[lon, nearest_energy_plant['lon']],
                mode='lines',
                line=dict(color='blue', width=2),
                hoverinfo='none'
            )
        )
        new_fig.add_trace(
            go.Scattermapbox(
                lat=[nearest_hospital['lat']],
                lon=[nearest_hospital['lon']],
                mode='markers',
                customdata=np.dstack((nearest_hospital['company_name'],
                                      nearest_hospital['phone_number'],
                                      nearest_hospital['number_of_employees'],
                                      nearest_hospital['previous_leaks_n'],
                                      nearest_hospital['fossil_fuel_type'],
                                      nearest_hospital['number_of_beds']
                                      )).T.tolist(),
                marker=dict(size=10, color='red'),
        hovertemplate='Company Name: %{customdata[0]} <br>Company telephone: %{customdata[1]} <br>Number of Employees: %{customdata[2]} <br>Number of Hospital Beds: % {customdata[5]}'
            )
        )

        new_fig.add_trace(
            go.Scattermapbox(
                lat=[nearest_energy_plant['lat']],
                lon=[nearest_energy_plant['lon']],
                mode='markers',
                customdata=np.dstack((nearest_hospital['company_name'],
                                      nearest_hospital['phone_number'],
                                      nearest_hospital['number_of_employees'],
                                      nearest_hospital['previous_leaks_n'],
                                      nearest_hospital['fossil_fuel_type'],
                                      nearest_hospital['number_of_beds']
                                      )),
                marker=dict(size=10, color='blue'),
                hovertemplate='Company Name: %{customdata[0]}<br>Company telephone: %{customdata[1]}<br>Number of Employees: %{customdata[2]}<br>Number of previous methane leaks: %{customdata[3]}<br>Fossil fuel type: %{customdata[4]}'
            )
        )

        # Set the zoom and view location from the previous figure
        new_fig.update_layout(mapbox=dict(center=dict(lat=fig2['layout']['mapbox']['center']['lat'],
                                                      lon=fig2['layout']['mapbox']['center']['lon']),
                                               zoom=fig2['layout']['mapbox']['zoom']))

        return new_fig

    return fig2

if __name__ == '__main__':
    app.run_server(host='localhost',port=8005)

z1, z2, z3 = np.random.random((3, 7, 7))

customdata = np.dstack((z2, z3))
mycustomdata = np.dstack((hospitals["company_name"], hospitals["number_of_employees"]))
mycustomdata = mycustomdata.T.tolist()


