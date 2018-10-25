#Static Image Export
#Need iPython notebook for this to work
import plotly

plotly.__version__

#Create a simple scatter plot with 100 random points of varying color and size
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.io as pio

import os
import numpy as np

init_notebook_mode(connected = True)

N = 100
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
sz = np.random.rand(N) * 30

fig = go.Figure()
fig.add_scatter(x = x,
                y = y,
                mode = 'markers',
                marker = {'size': sz,
                          'color': colors,
                          'opacity': 0.6,
                          'colorscale': 'Viridis'
                          });
iplot(fig)

#Write image file
if not os.path.exists('images'):
    os.mkdir('images')

pio.write_image(fig, 'images/fig1.png')
