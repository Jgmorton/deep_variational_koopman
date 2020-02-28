import gym
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load data
f = h5py.File('trial_data.h5', 'r')
states = f['x'][()]
actions = f['u'][()]
f.close()

# Define environment
env = gym.make('Pendulum-v0')
env = env.unwrapped
env.reset()

# Define array with black values
black_array = np.array([0, 0, 0], dtype=np.uint8)

# Define font for writing on image
fnt = ImageFont.truetype("/Library/Fonts/cmunbmo.ttf", 45)

# Loop through time steps
for t in range(0, 255, 4):
    # Initialize array to hold data from across all trials
    rgb_trials = np.zeros((2000, 2500, 3), dtype=np.uint8)

    # Loop through trials
    for i in range(4):
        for j in range(5):
            trial_num = i*5 + j
            full_state = states[trial_num, t]
            state = np.array([np.arctan2(full_state[1], full_state[0]), full_state[2]])
            env.set_state(state, actions[trial_num, t])
            rgb_pend = env.render(mode='rgb_array')[250:750, 250:750]
            rgb_pend[0, :] = black_array
            rgb_pend[499, :] = black_array
            rgb_pend[:, 0] = black_array
            rgb_pend[:, 499] = black_array

            # Convert image to array
            img = Image.fromarray(rgb_pend, 'RGB')
            d = ImageDraw.Draw(img)
            d.text((175,10), 'Trial ' + str(trial_num+1), font=fnt, fill=(0, 0, 0))
            rgb_pend = np.array(img)

            rgb_trials[i*500:(i+1)*500, j*500:(j+1)*500] = rgb_pend
    
    img = Image.fromarray(rgb_trials, 'RGB')
    img.save('pend_images/pend-' + str(t//4) + '.png')

# Now determine number of extra trials
extra_trials = states.shape[0] - 20
sets = extra_trials//2

# Loop through all sets and save images
for s in range(sets):
    for t in range(0, 255, 4):
        # Initialize array to hold data from across all trials
        rgb_trials = np.zeros((500, 1000, 3), dtype=np.uint8)

        # Loop through trials
        for i in range(2):
            trial_num = 20 + s*2 + i
            full_state = states[trial_num, t]
            state = np.array([np.arctan2(full_state[1], full_state[0]), full_state[2]])
            env.set_state(state, actions[trial_num, t])
            rgb_pend = env.render(mode='rgb_array')[250:750, 250:750]
            rgb_pend[0, :] = black_array
            rgb_pend[499, :] = black_array
            rgb_pend[:, 0] = black_array
            rgb_pend[:, 499] = black_array

            # Convert image to array
            img = Image.fromarray(rgb_pend, 'RGB')
            d = ImageDraw.Draw(img)
            d.text((175,10), 'Trial ' + str(trial_num+1), font=fnt, fill=(0, 0, 0))
            rgb_pend = np.array(img)

            rgb_trials[:, i*500:(i+1)*500] = rgb_pend
        
        img = Image.fromarray(rgb_trials, 'RGB')
        img.save('pend_images/set' + str(s) + '_t-' + str(t//4) + '.png')






