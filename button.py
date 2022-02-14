import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np


fig = plt.figure()
ICON_PLAY = plt.imread('https://i.stack.imgur.com/ySW6o.png')
ICON_PAUSE = plt.imread("https://i.stack.imgur.com/tTa3H.png")

# ICON_PLAY = cv.img2[X_line-self.padding[1]:X_line+1+self.padding[1], x-self.padding[0]:x+1+self.padding[0]]
def play(event):
    button_axes.images[0].set_data(ICON_PAUSE)
    fig.canvas.draw_idle()

button_axes = plt.axes([0.3, 0.3, 0.4, 0.4])
start_button = Button(button_axes, '', image=ICON_PLAY)
start_button.on_clicked(play)
plt.show()
