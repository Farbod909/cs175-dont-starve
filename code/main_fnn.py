import agent as ag
from replay import ReplayMemory
from helpers import state_as_one_hot

import sys
import os
import MalmoPython
import random
import time

from keras.models import Sequential
from keras.layers import Dense, InputLayer


if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

farm_size = 0
while farm_size % 2 == 0 or 3 > farm_size or 17 < farm_size:
    print("Input farm size (odd number, 17 max): ")
    farm_size = int(input())
farm_radius = int((farm_size-1)/2)

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Don't Starve</Summary>
              </About>
              
              <ServerSection>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,226*1;21;"/>
                  <DrawingDecorator>
                      <DrawCuboid x1="-9" y1="227" z1="-9" x2="9" y2="227" z2="9" type="air"/>
                      <DrawCuboid x1="-9" y1="226" z1="-9" x2="9" y2="226" z2="9" type="farmland"/>
                      <DrawBlock x="4" y="226" z="4" type="water"/>
                      <DrawBlock x="-4" y="226" z="4" type="water"/>
                      <DrawBlock x="4" y="226" z="-4" type="water"/>
                      <DrawBlock x="-4" y="226" z="-4" type="water"/>
                      <DrawBlock x="4" y="227" z="4" type="waterlily"/>
                      <DrawBlock x="-4" y="227" z="4" type="waterlily"/>
                      <DrawBlock x="4" y="227" z="-4" type="waterlily"/>
                      <DrawBlock x="-4" y="227" z="-4" type="waterlily"/>'''
missionXML +=        '<DrawLine x1="-' + str(farm_radius+1) + '" y1="227" z1="-' + str(farm_radius+1) + '" x2="' + str(farm_radius+1) + '" y2="227" z2="-' + str(farm_radius+1) + '" type="birch_fence"/>'
missionXML +=        '<DrawLine x1="-' + str(farm_radius+1) + '" y1="227" z1="-' + str(farm_radius+1) + '" x2="-' + str(farm_radius+1) + '" y2="227" z2="' + str(farm_radius+1) + '" type="birch_fence"/>'
missionXML +=        '<DrawLine x1="' + str(farm_radius+1) + '" y1="227" z1="' + str(farm_radius+1) + '" x2="' + str(farm_radius+1) + '" y2="227" z2="-' + str(farm_radius+1) + '" type="birch_fence"/>'
missionXML +=        '<DrawLine x1="' + str(farm_radius+1) + '" y1="227" z1="' + str(farm_radius+1) + '" x2="-' + str(farm_radius+1) + '" y2="227" z2="' + str(farm_radius+1) + '" type="birch_fence"/>'
missionXML +=    '''</DrawingDecorator>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>Wilson</Name>
                <AgentStart>
                    <Placement x="0.5" y="227" z="0.5" pitch="90" yaw="180"/>
                    <Inventory>
                        <InventoryItem slot="0" type="wheat_seeds" quantity="64"/>
                        <InventoryItem slot="1" type="carrot" quantity="64"/>
                        <InventoryItem slot="2" type="potato" quantity="64"/>
                        <InventoryItem slot="3" type="beetroot_seeds" quantity="64"/>
                    </Inventory>
                </AgentStart>
                <AgentHandlers>
                  <ContinuousMovementCommands/>
                  <DiscreteMovementCommands/>
                  <InventoryCommands/>
                  <ObservationFromFullStats/>
                  <ObservationFromGrid>
                      <Grid name="croplocal">
                        <min x="-2" y="0" z="-2"/>
                        <max x="2" y="1" z="2"/>
                      </Grid>
                      <Grid name="cropfull">'''
missionXML +=          '<min x="-' + str(farm_radius) + '" y="1" z="-' + str(farm_radius) + '"/>'
missionXML +=          '<max x="' + str(farm_radius) + '" y="1" z="' + str(farm_radius) + '"/>'
missionXML +=         '''</Grid>
                  </ObservationFromGrid>
                  <ObservationFromFullInventory/>
                  <ChatCommands/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

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


num_episodes = 1000

discount_rate = 0.95
exploration_rate = 0.5
max_exploration_rate = 0.5
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# define keras nn model here
model = Sequential()
model.add(InputLayer(batch_input_shape=(1, 54)))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(4, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])


reward_list = []

for episode in range(num_episodes):
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

    print()
    print("Mission running ", end=' ')

    agent_host.sendCommand("chat /time set day")

    agent = ag.Agent(agent_host)

    # Loop until mission ends:
    r_sum = 0
    while not agent.finished:
        state = agent.state.copy()
        if random.uniform(0, 1) > exploration_rate:
            # TODO: select action based on nn prediction
            action = np.argmax(model.predict(np.array([state_as_one_hot(state)])))
        else:
            action = agent.select_random_action()

        reward = agent.run(action)
        new_state = agent.state.copy()

        # TODO: update nn model weights here
        target = reward + discount_rate * \
            np.max(model.predict(np.array([state_as_one_hot(new_state)])))
        target_vec = model.predict(np.array([state_as_one_hot(state)]))[0]
        target_vec[action] = target
        model.fit(np.array([state_as_one_hot(state)]),
                  target_vec.reshape(-1, 4), epochs=1, verbose=False)


        r_sum += reward
    
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * \
        np.exp(-exploration_decay_rate * episode)
    reward_list.append(r_sum)

    print()
    print("Mission ended")
    agent_host.sendCommand("chat /kill @p")
    # Mission has ended.

print("reward_list:", reward_list)