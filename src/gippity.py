import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

# Fake data
x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
disp = np.sin(np.pi * x) * np.sin(np.pi * y)
stress = np.cos(np.pi * x) * np.cos(np.pi * y)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.3)  # Leave space for widget

cmap = ax.pcolormesh(x, y, disp, shading='auto')
cbar = plt.colorbar(cmap, ax=ax)
cbar.set_label('Displacement')

# Radio buttons
rax = plt.axes([0.05, 0.4, 0.2, 0.15])
radio = RadioButtons(rax, ('Displacement', 'Stress'))


def update(label):
    if label == 'Displacement':
        cmap.set_array(disp.ravel())
        cbar.set_label('Displacement')
    elif label == 'Stress':
        cmap.set_array(stress.ravel())
        cbar.set_label('Stress')
    cmap.changed()
    fig.canvas.draw_idle()


radio.on_clicked(update)
plt.show()
