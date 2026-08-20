"""Standalone script: animates the evolution of the fitted electron distribution function across
optimization iterations from a saved state_weights.txt, optionally saving it as a gif."""
import pickle
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.animation as animation

#copy mlflow output
#!aws s3 cp s3://public-ergodic-continuum/298/41dc3862aaa34b85972ff60786529206/artifacts/state_weights.txt ./retrived_run.txt

save_gif=True

# Path to a state_weights.txt produced by a fit run with optimizer.save_state enabled
# (see tsadar/inverse/loops.py). Edit this to point at the run you want to animate.
STATE_WEIGHTS_PATH = "state_weights.txt"

with open(STATE_WEIGHTS_PATH, "rb") as f:
    data = pickle.load(f)

length = len(data.keys())
save_points = np.empty(length)
ne = np.empty(length)
fe = np.empty([160,160,length])
for i,key in enumerate(data.keys()):
    save_points[i] = int(key)
    ne[i] = data[key]['species1']['ne']
    fe[:,:,i] = data[key]['species1']['fe']
    
ne = ne*(2.0-0.03)+0.03
fe = np.exp(fe*(-0.5+100)-100.0)

fig, ax = plt.subplots(1,2,figsize=(12, 5))

# ax.legend()
X,Y = np.meshgrid(np.linspace(-8.0,8.0,160),np.linspace(-8.0,8.0,160))
im1=ax[0].pcolormesh(X,Y,np.log(fe[:,:,0]),cmap='jet')
im2=ax[1].pcolormesh(X,Y,fe[:,:,0],cmap='jet')
time_text = ax[0].text(0.05, -0.1,'',horizontalalignment='left',verticalalignment='top', transform=ax[0].transAxes)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[0].set_xlabel('v_x/vth')
ax[1].set_xlabel('v_x/vth')
ax[0].set_ylabel('v_y/vth')
ax[1].set_ylabel('v_y/vth')
ax[0].set_title('ln(EDF) vs iteration')
ax[1].set_title('EDF vs iteration')

def update(frame):
    """FuncAnimation callback: updates the log(EDF)/EDF images and time label for the given frame index."""
    im1.set_array(np.log(fe[:,:,frame]))
    im2.set_array(fe[:,:,frame])
    time_text.set_text(f'time = {frame} / {length}')

    return im1, im2, time_text


ani = animation.FuncAnimation(fig=fig, func=update, frames=length, interval=20, blit=False)

if save_gif:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=15,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    ani.save('EDF90deg.gif', writer=writer)

plt.show()
