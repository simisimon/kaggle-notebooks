#!/usr/bin/env python
# coding: utf-8

# ### ⚡ This a faster version of [sazuma](https://www.kaggle.com/shoheiazuma)'s [Lux AI Imitation Learning Keras](https://www.kaggle.com/shoheiazuma/lux-ai-imitation-learning-keras).
# The difference is ≈ 640 seconds compared to 2096 seconds (train loop only time). Changes are made to `make_input` function and `__getitem__` in `LuxSequence` (inspired by similar speed up version of sazuma's Torch version of [imitation learning notebook](https://www.kaggle.com/shoheiazuma/lux-ai-with-imitation-learning) made by [Splend1dChan](https://www.kaggle.com/a24998667/fast-lux-ai-with-il-cached-inputs).

# In[ ]:


get_ipython().system('pip install kaggle-environments -U > /dev/null 2>&1')
get_ipython().system('cp -r ../input/lux-ai-2021/* .')


# In[ ]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import math
import json
from pathlib import Path
import random
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# In[ ]:


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)

seed = 42
seed_everything(seed)


# # Preprocessing

# In[ ]:


def to_label(action):
    strs = action.split(' ')
    unit_id = strs[1]
    if strs[0] == 'm':
        label = {'c': None, 'n': 0, 's': 1, 'w': 2, 'e': 3}[strs[2]]
    elif strs[0] == 'bcity':
        label = 4
    else:
        label = None
    return unit_id, label


def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True


def create_dataset_from_json(episode_dir, team_name='Toad Brigade'): 
    obses = {}
    x = []
    y = []
    
    episodes = list(Path(episode_dir).glob('*[0-9].json'))
    for filepath in tqdm(episodes): 
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        index = np.argmax([r or 0 for r in json_load['rewards']])
        if json_load['info']['TeamNames'][index] != team_name:
            continue

        for i in range(len(json_load['steps'])-1):
            if json_load['steps'][i][index]['status'] == 'ACTIVE':
                actions = json_load['steps'][i+1][index]['action']
                obs = json_load['steps'][i][0]['observation']
                
                if depleted_resources(obs):
                    break
                
                obs['player'] = index
                obs = dict([
                    (k,v) for k,v in obs.items() 
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])
                obs_id = f'{ep_id}_{i}'
                obses[obs_id] = obs
                                
                for action in actions:
                    unit_id, label = to_label(action)
                    if label is not None:
                        x.append((obs_id, unit_id))
                        y.append(label)

    return obses, x, y


# In[ ]:


episode_dir = '../input/lux-ai-episodes'
obses, x, y = create_dataset_from_json(episode_dir)
print('Observations:', len(obses), 'Samples:', len(x))


# In[ ]:


actions = ['Move North', 'Move South', 'Move West', 'Move East', 'Build City']
for value, count in zip(*np.unique(y, return_counts=True)):
    print(f'{actions[value]:<11}: {count:>3}')


# # Training

# In[ ]:


# Input for Neural Network
def make_uid2pos(obs):
    ret = {}
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        if input_identifier == 'u':
            x = int(strs[4]) 
            y = int(strs[5]) 
            ret[strs[3]] = (x,y)
    return ret

def make_array(obs):
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}
    
    b = np.zeros((32, 32, 20), dtype=np.float32)
    
    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]
        
        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            team = int(strs[2])
            cooldown = float(strs[6])
            idx = 2 + (team - obs['player']) % 2 * 3
            b[x, y, idx:idx + 3] = (
                1,
                cooldown / 6,
                (wood + coal + uranium) / 100
            )
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            idx = 8 + (team - obs['player']) % 2 * 2
            b[x, y, idx:idx + 2] = (
                1,
                cities[city_id]
            )
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(strs[4])
            b[x, y, {'wood': 12, 'coal': 13, 'uranium': 14}[r_type]] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[:, :, 15 + (team - obs['player']) % 2] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10
    
    # Day/Night Cycle
    b[:, :, 17] = obs['step'] % 40 / 40
    # Turns
    b[:, :, 18] = obs['step'] / 360
    # Map Size
    b[x_shift:32 - x_shift, y_shift:32 - y_shift, 19] = 1

    return b


class LuxSequence(Sequence):
    def __init__(self, obses, x_set, y_set, batch_size):
        self.obses = obses
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.obses_array = {k: make_array(obs) for k, obs in obses.items()}
        self.uid2pos = {k: make_uid2pos(obs) for k, obs in obses.items()}
        self.obs_size = {k: obs['width'] for k, obs in obses.items()}
        del self.obses
        
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[
            idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[
            idx * self.batch_size:(idx + 1) * self.batch_size]
        
        states = []
        for ind in range(len(batch_x)):
            obs_id, unit_id = batch_x[ind]
            unit_pos = self.uid2pos[obs_id][unit_id]
            size = self.obs_size[obs_id]
            state = self.obses_array[obs_id]
            
            shift = (32 - size) // 2
            x = unit_pos[0] + shift
            y = unit_pos[1] + shift
            state[:, :, :2] = 0
            state[x, y, 0] = state[x, y, 2]
            state[x, y, 1] = state[x, y, 4]
            states.append(state)
            
        states = np.array(states)        
        actions = np.array(batch_y)
        return states, actions


# In[ ]:


# Neural Network for Lux AI
def create_luxnet(blocks=12, filters=32):
    inputs = layers.Input((32, 32, 20))
    x = layers.Conv2D(filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    for _ in range(blocks):
        residual = layers.Conv2D(filters, 3, padding='same')(x)
        residual = layers.BatchNormalization()(residual)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)

    x = tf.reduce_sum(x * inputs[:,:,:,:1], [1, 2])
    outputs = layers.Dense(5, activation='softmax')(x)    
    
    model = Model(inputs, outputs)
    return model


# In[ ]:


model = create_luxnet()
model.summary()


# In[ ]:


model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)    
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)
batch_size = 64
train_seq = LuxSequence(obses, x_train, y_train, batch_size)
val_seq = LuxSequence(obses, x_val, y_val, batch_size)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='model.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)


# In the original code by [sazuma](https://www.kaggle.com/shoheiazuma/lux-ai-imitation-learning-keras), one epoch took approximately 140 seconds

# In[ ]:


history = model.fit(
    train_seq, 
    validation_data=val_seq,
    callbacks=[model_checkpoint],
    epochs=15,
    workers=2,
    use_multiprocessing=True
)


# In[ ]:


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

epochs = range(len(accuracy))

plt.figure()

plt.plot(epochs, accuracy, label='Training accuracy')
plt.plot(epochs, val_accuracy, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')

plt.show()


# # Submission

# In[ ]:


get_ipython().run_cell_magic('writefile', 'agent.py', 'import os\nimport numpy as np\nimport tensorflow as tf\nfrom lux.game import Game\n\n\npath = \'/kaggle_simulations/agent\' if os.path.exists(\'/kaggle_simulations\') else \'.\'\nmodel = tf.keras.models.load_model(f\'{path}/model.h5\')\n\n\ndef make_input(obs, unit_id):\n    width, height = obs[\'width\'], obs[\'height\']\n    x_shift = (32 - width) // 2\n    y_shift = (32 - height) // 2\n    cities = {}\n    \n    b = np.zeros((32, 32, 20), dtype=np.float32)\n    \n    for update in obs[\'updates\']:\n        strs = update.split(\' \')\n        input_identifier = strs[0]\n        \n        if input_identifier == \'u\':\n            x = int(strs[4]) + x_shift\n            y = int(strs[5]) + y_shift\n            wood = int(strs[7])\n            coal = int(strs[8])\n            uranium = int(strs[9])\n            if unit_id == strs[3]:\n                # Position and Cargo\n                b[x, y, :2] = (\n                    1,\n                    min(wood + coal + uranium, 100) / 100\n                )\n            # Units\n            team = int(strs[2])\n            cooldown = float(strs[6])\n            idx = 2 + (team - obs[\'player\']) % 2 * 3\n            b[x, y, idx:idx + 3] = (\n                1,\n                cooldown / 6,\n                min(wood + coal + uranium, 100) / 100\n            )\n        elif input_identifier == \'ct\':\n            # CityTiles\n            team = int(strs[1])\n            city_id = strs[2]\n            x = int(strs[3]) + x_shift\n            y = int(strs[4]) + y_shift\n            idx = 8 + (team - obs[\'player\']) % 2 * 2\n            b[x, y, idx:idx + 2] = (\n                1,\n                cities[city_id]\n            )\n        elif input_identifier == \'r\':\n            # Resources\n            r_type = strs[1]\n            x = int(strs[2]) + x_shift\n            y = int(strs[3]) + y_shift\n            amt = int(strs[4])\n            b[x, y, {\'wood\': 12, \'coal\': 13, \'uranium\': 14}[r_type]] = amt / 800\n        elif input_identifier == \'rp\':\n            # Research Points\n            team = int(strs[1])\n            rp = int(strs[2])\n            b[:, :, 15 + (team - obs[\'player\']) % 2] = min(rp, 200) / 200\n        elif input_identifier == \'c\':\n            # Cities\n            city_id = strs[2]\n            fuel = float(strs[3])\n            lightupkeep = float(strs[4])\n            cities[city_id] = min(fuel / lightupkeep, 10) / 10\n    \n    # Day/Night Cycle\n    b[:, :, 17] = obs[\'step\'] % 40 / 40\n    # Turns\n    b[:, :, 18] = obs[\'step\'] / 360\n    # Map Size\n    b[x_shift:32 - x_shift, y_shift:32 - y_shift, 19] = 1\n\n    return b\n\n\ngame_state = None\ndef get_game_state(observation):\n    global game_state\n    \n    if observation["step"] == 0:\n        game_state = Game()\n        game_state._initialize(observation["updates"])\n        game_state._update(observation["updates"][2:])\n        game_state.id = observation["player"]\n    else:\n        game_state._update(observation["updates"])\n    return game_state\n\n\ndef in_city(pos):    \n    try:\n        city = game_state.map.get_cell_by_pos(pos).citytile\n        return city is not None and city.team == game_state.id\n    except:\n        return False\n\n\ndef call_func(obj, method, args=[]):\n    return getattr(obj, method)(*args)\n\n\nunit_actions = [(\'move\', \'n\'), (\'move\', \'s\'), (\'move\', \'w\'), (\'move\', \'e\'), (\'build_city\',)]\ndef get_action(policy, unit, dest):\n    for label in np.argsort(policy)[::-1]:\n        act = unit_actions[label]\n        pos = unit.pos.translate(act[-1], 1) or unit.pos\n        if pos not in dest or in_city(pos):\n            return call_func(unit, *act), pos \n            \n    return unit.move(\'c\'), unit.pos\n\n\ndef agent(observation, configuration):\n    global game_state\n    \n    game_state = get_game_state(observation)    \n    player = game_state.players[observation.player]\n    actions = []\n    \n    # City Actions\n    unit_count = len(player.units)\n    for city in player.cities.values():\n        for city_tile in city.citytiles:\n            if city_tile.can_act():\n                if unit_count < player.city_tile_count: \n                    actions.append(city_tile.build_worker())\n                    unit_count += 1\n                elif not player.researched_uranium():\n                    actions.append(city_tile.research())\n                    player.research_points += 1\n    \n    # Worker Actions\n    units = [\n        unit for unit in player.units\n        if unit.can_act() and (game_state.turn % 40 < 30 or not in_city(unit.pos))\n    ]\n    states = np.array([make_input(observation, unit.id) for unit in units])\n    \n    if len(states) > 0:\n        policies = model.predict(states)\n\n        dest = []\n        for policy, unit in zip(policies, units):\n            action, pos = get_action(policy, unit, dest)\n            actions.append(action)\n            dest.append(pos)\n\n    return actions\n')


# In[ ]:


from kaggle_environments import make

env = make("lux_ai_2021", configuration={"width": 12, "height": 12, "loglevel": 1, "annotations": True}, debug=True)
steps = env.run(['agent.py', 'agent.py'])
env.render(mode="ipython", width=1200, height=800)


# In[ ]:


get_ipython().system('tar -czf submission.tar.gz *')

