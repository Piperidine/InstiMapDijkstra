# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Output, Input
import dash_bootstrap_components as dbc


import json
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import math

class Graph():
  
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] 
                    for row in range(vertices)]

    def minDistance(self, dist, sptSet):
        min = float('inf')
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
  
        return min_index
    def dijkstra(self, src):
  
        dist = [float('inf')] * self.V
        prev = [None] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
  
        for cout in range(self.V):
            u = self.minDistance(dist, sptSet)

            sptSet[u] = True
            for v in range(self.V):
                if self.graph[u][v] > 0 and sptSet[v] == False and \
                dist[v] > dist[u] + self.graph[u][v]:
                        dist[v] = dist[u] + self.graph[u][v]
                        prev[v] = u
        return dist, prev

df = pd.read_csv('distances.csv')[['Name','lat','long','Number']]

px.set_mapbox_access_token('pk.eyJ1IjoiYW50b3JudmF5IiwiYSI6ImNrbmsyb3M3NDA3NGUycHM1bzg0MXBmeGwifQ.XDoxAtNYH_JlmZz0e7ZsnQ')
fig = px.scatter_mapbox(df, lat="lat", lon="long",mapbox_style="satellite", zoom=14, color_discrete_sequence=['red']*len(df), labels=None, hover_name='Name', hover_data={'lat':False, 'long':False}, )
fig.update_layout(showlegend=False)
fig.update_layout(
    autosize=True,
    height=800,)

edges = [
    [1,5],
    [5,4],
    [4,2],
    [2,3],
    [3,37],
    [37,18],
    [18,17],
    [17,19],
    [19,20],
    [20,22],
    [22,23],
    [23,24],
    [20,21],
    [19,25],
    [25,27],
    [27,26],
    [26,28],
    [28,29],
    [29,30],
    [30,31],
    [31,32],
    [31,33],
    [33,34],
    [34,35],
    [35,40],
    [40,41],
    [41,42],
    [42,43],
    [43,44],
    [44,45],
    [2,45],
    [44,46],
    [37,36],
    [36,30],
    [36,38],
    [38,35],
    [38,39],
    [17,16],
    [16,14],
    [14,15],
    [14,13],
    [13,12],
    [12,9],
    [9,10],
    [10,11],
    [7,12],
    [8,9],
    [6,8],
    [8,7],
    [4,7],
    [5,6],
]

def distance(lat1,lon1,lat2,lon2):
    R = 6373.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance * 1000

g = Graph(len(df))

for i in edges:
    x = i[0]
    y = i[1]
    d = distance(df.iloc[x-1].lat,df.iloc[x-1].long, df.iloc[y-1].lat,df.iloc[y-1].long)
    g.graph[x-1][y-1] = d
    g.graph[y-1][x-1] = d

from flask import Flask

server = Flask(__name__)
app = dash.Dash(server=server,external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div(style={ 'background-color':'white', 'height':'100%'
    
},
    children=[
        html.H1(children="Dijkstra's X Insti", style={'margin-top':'2.5%', 'text-align':'center'}),
        dbc.Row([
        dbc.Col(md= 2),
        dbc.Col([
            dcc.Dropdown(
                id='crossfilter-start',
                options=[{'label': i[1].Name, 'value': i[1].Number} for i in df.iterrows()],
                value=1,
                placeholder='Select a start'

            ),
        ],
            md=3),
        dbc.Col(md= 2),
        dbc.Col([
            dcc.Dropdown(
                id='crossfilter-dest',
                options=[{'label': i[1].Name, 'value': i[1].Number} for i in df.iterrows()],
                value=None,
                placeholder='Select a destination'

            ),
        ],
            md=3),
        ]),
        dbc.Row([
            dbc.Col(md=2),
            dbc.Col(
            dcc.Graph(
                id='crossfilter-insti',
                figure=fig,
                # layout=fig.layout,
            ), md=8),
        ],
            ),
    ])

@app.callback(
    Output('crossfilter-insti', 'figure'),
    [Input('crossfilter-start', 'value'),
    Input('crossfilter-dest', 'value')])
def update_graph(start,destination):
    fig.data = [fig.data[0]]

    if start is None or destination is None:
        return fig
    
    start-=1
    destination-=1   

    g = Graph(len(df))
    for i in edges:
        x = i[0]
        y = i[1]
        d = distance(df.iloc[x-1].lat,df.iloc[x-1].long, df.iloc[y-1].lat,df.iloc[y-1].long)
        g.graph[x-1][y-1] = d
        g.graph[y-1][x-1] = d
    dist, prev = g.dijkstra(start)

    path = [destination]
    src = start
    while True:
        dest = prev[path[-1]]
        if dest is None:
            break
        else:
            path.append(dest)
    path = [i+1 for i in path[::-1]]
    visible_edges = []
    for i in range(len(path)-1):
        try:
            visible_edges.append(edges.index([path[i],path[i+1]]))
        except:
            visible_edges.append(edges.index([path[i+1],path[i]]))
    
    for j in visible_edges:
        i = edges[j]
        x = i[0]
        y = i[1]
        
        fig.add_trace(go.Scattermapbox(
        mode = "lines",
        lon = [df.iloc[x-1].long, df.iloc[y-1].long],
        lat = [df.iloc[x-1].lat,df.iloc[y-1].lat],
        hoverinfo='skip',
        line={'color':'orange'}
        ))

    return fig

if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
