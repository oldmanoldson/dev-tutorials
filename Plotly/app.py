#Plotly Dash Template
#Import libraries
import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly
plotly.__version__

import plotly.plotly as py
import plotly.graph_objs as go

import numpy as np

#Import, clean, and setup data
N = 100
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
sz = np.random.rand(N) * 30

#Begin Dashboard here
#Styling
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets = external_stylesheets)

#Build the dashboard app
#In the dcc.Graph function use the same syntax as found in plotly for python
app.layout = html.Div(children = [
    html.H1(children = 'Main Title'),

    html.Div(children = '''
        Subtitle
    '''),

    dcc.Graph(
        id = 'example-graph',
        figure = go.Figure(
            data = [go.Scatter(x = x, y = y, mode = 'markers', marker = {'size': sz, 'color': colors, 'opacity': 0.6, 'colorscale': 'Viridis'})],
            layout = go.Layout(title = 'Graph Title')
        )
    )
])

if __name__ == '__main__':
    app.run_server(debug = True)
