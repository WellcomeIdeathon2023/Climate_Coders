from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd

import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

#import cdstoolbox as ct
#import chart_studio.plotly as py

import chart_studio.plotly as py


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
final_daily2 = pd.read_csv('C:/Users/bengo/OneDrive - Nexus365/Documents/Climate_coders_ben_branch/Climate_Coders/data/combined_daily.csv')
# Round 'lon_m' and 'lat_m' columns
final_daily2['lon_m'] = final_daily2['lon_m'].round()
final_daily2['lat_m'] = final_daily2['lat_m'].round()
# Convert 'time' column to datetime
final_daily2['date'] = pd.to_datetime(final_daily2['time']).dt.date
# Group by 'date', 'lon_m', and 'lat_m', and calculate the mean of 'ch4'
mean_ch4 = final_daily2.groupby(['date', 'lon_m', 'lat_m'])['ch4'].mean().reset_index()
# Remove rows with missing values
mean_ch4 = mean_ch4.dropna()


import plotly.express as px
# Create a Plotly choropleth map
fig2 = px.scatter_mapbox(mean_ch4,
                        lat='lat_m',
                        lon='lon_m',
                        color='ch4',
                        color_continuous_scale='viridis',
                        range_color=(mean_ch4['ch4'].min(), mean_ch4['ch4'].max()),
                        mapbox_style='open-street-map',
                        zoom=3,
                        center={'lat': 39.8283, 'lon': -98.5795},
                        hover_data=['date'],
                        labels={'ch4': 'ppb'})
# Update plot layout
fig2.update_layout(title='Methane emissions over the United States, 2019',
                  coloraxis_colorbar=dict(title='ppb'),
                  legend=dict(orientation='h', yanchor='top', y=1, xanchor='right', x=1))


####Dashboard####
app = Dash(__name__)

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


app.layout = html.Div([
    dcc.Tabs(id="tabs-styled-with-inline", value='tab-1', children=[
        dcc.Tab(label='Methane Map', value='tab-1', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Methane Graph', value='tab-2', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Health Map', value='tab-3', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Health Graph', value='tab-4', style=tab_style, selected_style=tab_selected_style),
    ], style=tabs_styles),
    html.Div(id='tabs-content-inline')
])

@callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('Map of Methane Levels, overlayed with energy plant and hospital locations'),
                            dcc.Graph( id='Methane Stripes', figure=fig2)

        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Methane data visualisations'),
                dcc.Graph( id='Methane Stripes', figure=fig)
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Map of Mortalities from respiratory diseases and gas leaks')
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Health data visualisations')
        ])



if __name__ == '__main__':
    app.run_server(host='localhost',port=8005)














