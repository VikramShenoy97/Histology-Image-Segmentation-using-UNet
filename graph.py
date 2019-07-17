import os
import pandas as pd
import math
import numpy as np
from PIL import Image

from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go

filename = "Saved_Model/history.csv"
# Open the history file that stores the training accuracy and loss.
history = pd.read_csv(filename, header=0, low_memory=False)
history_array = history.values
epochs = history_array[:, 0]
training_accuracy = history_array[:, 1]
training_loss = history_array[:, 2]

py.sign_in('VikramShenoy','x1Un4yD3HDRT838vRkFA')

# Generate accuracy and loss trace.
trace0 = go.Scatter(
x = epochs,
y = training_accuracy,
mode = "lines",
name = "Training Accuracy"
)

trace1 = go.Scatter(
x = epochs,
y = training_loss,
mode = "lines",
name = "Training Loss"
)
# Create Training Accuracy vs Loss Graph
data = go.Data([trace0, trace1])
layout = go.Layout(title="Training Accuracy vs Loss")
fig = go.Figure(data=data, layout=layout)
fig['layout']['xaxis'].update(title="Number of Epochs", range = [min(epochs), max(epochs)], dtick=len(epochs)/10, showline = True, zeroline=True,  mirror='ticks', linecolor='#636363', linewidth=2)
fig['layout']['yaxis'].update(title="Training Accuracy / Loss", range = [0, 1], dtick=0.1, showline = True, zeroline=True, mirror='ticks',linecolor='#636363',linewidth=2)
py.image.save_as(fig, filename="Training_Graph.png")

print "Training Graph Created"
