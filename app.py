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

from constants import Graph, edges, distance

from grapher import ret_fig

df = pd.read_csv('distances.csv')[['Name','lat','long','Number']]

px.set_mapbox_access_token('pk.eyJ1IjoiYW50b3JudmF5IiwiYSI6ImNrbmsyb3M3NDA3NGUycHM1bzg0MXBmeGwifQ.XDoxAtNYH_JlmZz0e7ZsnQ')
fig = px.scatter_mapbox(df, lat="lat", lon="long",mapbox_style="satellite", zoom=14, color_discrete_sequence=['red']*len(df), labels=None, hover_name='Name', hover_data={'lat':False, 'long':False}, )
fig.update_layout(showlegend=False)
fig.update_layout(
    autosize=True,
    height=800,)


g = Graph(len(df))
dijktime = 0
for i in edges:
    x = i[0]
    y = i[1]
    d = distance(df.iloc[x-1].lat,df.iloc[x-1].long, df.iloc[y-1].lat,df.iloc[y-1].long)
    g.graph[x-1][y-1] = d
    g.graph[y-1][x-1] = d

from flask import Flask

server = Flask(__name__)
app = dash.Dash(server=server,external_stylesheets=[dbc.themes.BOOTSTRAP])
imap = [
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
            dbc.Col([
                dbc.Button("Go to Graph Generator", color="dark",href='/network', style={'margin-left':'50%'}),
            ], 
            md=2),
            dbc.Col(
            dcc.Graph(
                id='crossfilter-insti',
                figure=fig,
                # layout=fig.layout,
            ), md=8),
        ],
            ),
    ]

grap = [
    html.H3('Random Geometric Graphs', style={'margin-top':'2.5%', 'text-align':'center'}),
    dcc.Dropdown(
        id='crossfilter-n',
        options=[{'label': i, 'value': i} for i in [5,10,25,50,100,200,300,500,1000]],
        value=50,
        placeholder='Select a destination',
        style={'width':'40%', 'margin-left':'30%'},
        ),
    html.Div(className='row',
    children=[
        html.Div([
            dbc.Button("Go to InstiMap", color="dark",href='/', style={'margin-left':'30%'}),
            html.Table([
                html.Tr([
                    html.Th('Algorithm'),
                    html.Th('Time Taken (ms)'),
                    html.Th('Relative'),
                ])
            ],style={'width':'100%', 'margin-left':'5%','margin-top':'15%'}, id='time-table')
        ], className='col-md-3'),
        html.Div([
            dcc.Graph(id='main-graph', figure=fig)
        ], className='col-md-9')
    ])
]

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content', style={ 'background-color':'white', 'height':'100%'})
])

app.validation_layout = html.Div([
    url_bar_and_content_div,
    imap,
    grap])

app.layout = url_bar_and_content_div

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')])
def display_page(pathname):
    if pathname in ['/network/','/network']:
        return grap
    elif pathname == '/':
        return imap
    else:
        return imap

@app.callback(
    Output('main-graph', 'figure'),
    Output('time-table', 'children'),
    [Input('crossfilter-n', 'value')])
def display_fig(n):
    if n is None:
        return [go.Figure(data=[]),[]]
    figu, timeit = ret_fig(n)
    timeit = timeit[0][0]
    if figu is None:
        figu = go.Figure(data=[], layout=go.Layout(
            annotations=[ dict(
                    text="Figure not loaded to prevent overload",
                    showarrow=False,
                    xref="paper", yref="paper",
                    font_size=30,
                    x=0.5, y=0.5 ) ],
        ))
        l = [('Dijkstra\'s',timeit[0]),('A*',timeit[2])]
        l.sort(key=lambda x:x[1])
        lay = [
                html.Tr([
                    html.Th('Algorithm'),
                    html.Th('Time Taken (ms)'),
                    html.Th('Relative'),
                ]),
                html.Tr([
                    html.Th('{}'.format(l[0][0])),
                    html.Th('{}'.format(int(l[0][1]*100000)/100)),
                    html.Th('1x'),
                ]),
                html.Tr([
                    html.Th('{}'.format(l[1][0])),
                    html.Th('{}'.format(int(l[1][1]*100000)/100)),
                    html.Th('~{}x'.format(int(l[1][1]/l[0][1]))),
                ]),
                ]

        return [figu,lay]
    else:
        l = [('Dijkstra\'s',timeit[0]),('Bellman-Ford',timeit[1]),('A*',timeit[2])]
        l.sort(key=lambda x:x[1])
        lay = [
                html.Tr([
                    html.Th('Algorithm'),
                    html.Th('Time Taken (ms)'),
                    html.Th('Relative'),
                ]),
                html.Tr([
                    html.Th('{}'.format(l[0][0])),
                    html.Th('{}'.format(int(l[0][1]*100000)/100)),
                    html.Th('1x'),
                ]),
                html.Tr([
                    html.Th('{}'.format(l[1][0])),
                    html.Th('{}'.format(int(l[1][1]*100000)/100)),
                    html.Th('~{}x'.format(int(l[1][1]/l[0][1]))),
                ]),
                html.Tr([
                    html.Th('{}'.format(l[2][0])),
                    html.Th('{}'.format(int(l[2][1]*100000)/100)),
                    html.Th('~{}x'.format(int(l[2][1]/l[0][1]))),
                ]),
                ]
        return [figu,lay]

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
    dist, prev, t_taken = g.dijkstra(start)
    # print(t_taken.microseconds/1000)
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
    app.run_server(host='0.0.0.0', debug=False)
