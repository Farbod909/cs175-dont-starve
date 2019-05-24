import agent
import conv_net as cnn
from replay import ReplayMemory

import sys
import os
import MalmoPython
import time


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

net = cnn.Net()
memory = ReplayMemory(1000)

for episode in range(5):
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

    agent = agent.Agent(agent_host)

    # Loop until mission ends:
    while not agent.finished:
        state = agent.state.copy()
        action = agent.select_action(state)
        reward = agent.run(action)
        memory.push(state, action, agent.state.copy(), reward)
        net.train(memory)
                    
    print(agent.state)   
    memory.print_replay()
    #net.forward(agent.state)

    print()
    print("Mission ended")
    agent_host.sendCommand("chat /kill @p")
    # Mission has ended.
