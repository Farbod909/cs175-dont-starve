import agent as ag
from replay import ReplayMemory, Transition
from env_xml import generate_env

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, concatenate

import sys
import os
import MalmoPython
import time
import numpy as np
import copy
import pickle


if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)


def transform_farm(farm_array):
    """
    This function turns the array into a format that convolutional layers can work with. In this case, we get the farm array to mimick the format of images which have 3 dimensions: [channels, height, width]
    """
    side_len = int(np.sqrt(len(farm_array)))
    farm_mat = np.array(farm_array,dtype=float).reshape(side_len,side_len) #converts array into square matrix
    farm_mat = np.expand_dims(farm_mat, axis=0) #adds an extra dimension at the index specified. 
    return farm_mat

def select_action(state, net = None):
    import random
    eps_threshold = exploration_rate
    sample = random.random()

    if net != None and sample > eps_threshold: #for now, this guarantees random action everytime
        prediction = net.predict(np.expand_dims(state, axis=0))
        return prediction.argmax() 
            
    else:
        return random.randint(0,1)

def dec_exploration_rate(episode):
    return min_exploration + (max_exploration-min_exploration) * np.exp(-exploration_decay_rate * episode)
    

discount_rate = .45
exploration_rate = .5
max_exploration = .3 
min_exploration = .01
exploration_decay_rate = .001

missionXML = generate_env()

# Create default Malmo objects:
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)



#create model

from keras.models import load_model
net = load_model('cnn.h5')

#net = Sequential()

#model layers
#net.add(Conv2D(6, kernel_size=2, activation = 'relu', input_shape=(1,3,3), data_format='channels_first'))
#net.add(Conv2D(12, kernel_size=2, activation = 'relu')) 
#net.add(Flatten())
#net.add(Dense(90, activation='tanh'))
#net.add(Dense(30, activation='tanh'))
#net.add(Dense(4, activation='softmax'))
#net.compile(optimizer='adam', loss = 'mse', metrics= ["mae"] )


import pickle
f = open('memory.txt','rb')
memory = pickle.load(f)

#memory = ReplayMemory(1000000)
#rewards = open("rewards.txt", 'w')
#rewards.close()

for episode in range(40000):
    my_mission = MalmoPython.MissionSpec(missionXML, True)
    my_mission_record = MalmoPython.MissionRecordSpec()

    # Attempt to start a mission:
    max_retries = 3
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_mission_record )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

    # Loop until mission starts:
    print("Waiting for the mission to start ", end=' ')
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print("Mission running ", end=' ')

    agent_host.sendCommand("chat /time set day")
    agent = ag.Agent(agent_host)

    # Loop until mission ends:
    while not agent.finished:
        state = transform_farm(copy.deepcopy(agent.state))
        action = select_action(state, net) 
        reward = np.array(agent.run(action+1))
        memory.push(state, action, transform_farm(copy.deepcopy(agent.state)), reward)

        sample = memory.sample(1)[0]

        

        target = sample.reward + discount_rate * \
                np.max(net.predict(np.expand_dims(sample.next_state, axis=0)))
        target_vec = net.predict(np.expand_dims(sample.state, axis=0))[0]
        target_vec[sample.action] = target
        net.fit(np.expand_dims(sample.state, axis=0), np.expand_dims(target_vec, axis=0), epochs=20,verbose=0)
        dec_exploration_rate(episode)               


    with open("rewards.txt", 'a') as r:
        r.write('%1.2f\n'%reward)


    if episode % 1000 == 0:
        net.save('cnn.h5')
        with open("memory.txt", 'wb') as f:
            pickle.dump(memory, f)

    print()
    print("Mission ended")
    agent_host.sendCommand("chat /kill @p")
    # Mission has ended.
