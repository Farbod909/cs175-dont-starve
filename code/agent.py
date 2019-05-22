from __future__ import print_function
from builtins import range
import MalmoPython
import os
import sys
import time
import json
import random

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

def getWorldState():
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    while len(world_state.observations) == 0: #wait for the first observations
        time.sleep(.1)
        world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)
        
    msg = world_state.observations[-1].text
    return json.loads(msg)

def chooseDirection(direction, grid):
    if direction == "east" and grid[13] == "birch_fence":
        if grid[17] == "birch_fence":
            return "nextstage"
        return "south"
    elif direction == "west" and grid[11] == "birch_fence":
        if grid[17] == "birch_fence":
            return "nextstage"
        return "south"
    elif direction == "south":
        if grid[11] == "birch_fence":
            return "east"
        else:
            return "west"
    return direction

def updateState(state, crop, direction):
    pos = state.index(-1)
    state[pos] = crop
    if direction == "east":
        state[pos+1] = -1
    elif direction == "west":
        state[pos-1] = -1
    elif direction == "south":
        state[pos+farm_size] = -1
    return state

def stage0(grid): #move to corner
    if grid[7] == "air":
        agent_host.sendCommand("movenorth 1")
    elif grid[11] == "air":
        agent_host.sendCommand("movewest 1")
    else:
        return 1
    return 0

def stage1(grid, direction, crop, state): #plant crops
    if grid[12] != "waterlily":
        agent_host.sendCommand("hotbar." + str(crop) + " 1")
        agent_host.sendCommand("hotbar." + str(crop) + " 0")
        agent_host.sendCommand("use 1")
        time.sleep(0.1)

    direction = chooseDirection(direction, grid)
    if direction == "nextstage":
        return 2, direction, reward, state
            
    agent_host.sendCommand("move"+direction+" 1")
    state = updateState(state, crop, direction)
    return 1, direction, 0, state

def stage2(): #grow crops
    agent_host.sendCommand("chat /tp 0 ~ 0")
    time.sleep(.5)
    agent_host.sendCommand("chat /gamerule randomTickSpeed 10000")
    time.sleep(1)
    agent_host.sendCommand("chat /gamerule randomTickSpeed 1")
    return 3

def stage3(observations): #harvest crops
    full_grid = observations.get(u'cropfull', 0)
    print("Full grid: " + str(full_grid))
    agent_host.sendCommand("hotbar.9 1") #for some reason using chat commands causes the agent to right click
    agent_host.sendCommand("hotbar.9 0")
    agent_host.sendCommand('chat /fill -' + str(farm_radius) + ' 227 -' + str(farm_radius) + ' ' + str(farm_radius) + ' 227 ' + str(farm_radius) + ' air 0 destroy') #this needs to vary if the size of the field changes
    time.sleep(2)
    agent_host.sendCommand("chat /tp @e[type=item] @p")
    time.sleep(2)
    return 4

def stage4(observations): #count crops
    wheat, carrot, potato, beetroot = 0, 0, 0, 0
    for i in range(0,41):
        item = observations.get(u'InventorySlot_'+str(i)+'_item', 0)
        if item == "wheat":
            wheat += observations.get(u'InventorySlot_'+str(i)+'_size', 0)
        elif item == "carrot":
            carrot += observations.get(u'InventorySlot_'+str(i)+'_size', 0)
        elif item == "potato":
            potato += observations.get(u'InventorySlot_'+str(i)+'_size', 0)
        elif item == "beetroot":
            beetroot += observations.get(u'InventorySlot_'+str(i)+'_size', 0)
    #if initial resources change, the 64's below need to change as well
    print("\nHarvested " + str(wheat) + " wheat, " + str(carrot-64) + " carrots, " + str(potato-64) + " potatoes, " + str(beetroot) + " beetroots.")

    #subtract the initial carrots and potatoes, and account for harvesting multiples at random from grown crops
    reward = wheat + (carrot - 64) / 1.71 + (potato - 64) / 1.71 + beetroot
    print("Reward is "+str(reward) + " out of " + str(farm_size**2)) #can technically be higher with good RNG
    return 5, reward



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
stage = 0
direction = "east"
planting = False
state = [0] * farm_size**2
state[0] = -1

# Loop until mission ends:
while stage < 5:
    observations = getWorldState()
    grid = observations.get(u'croplocal', 0)
    #because farmland is not a full block, the grid observations sometimes include the ground, so we remove it
    if grid[0] == "stone" or grid[0] == "dirt" or grid[0] == "farmland" or grid[0] == "water":
        del grid[0:25]
       
    if stage == 0: #move to corner and start planting
        stage = stage0(grid)

    elif stage == 1: #planting
        crop = random.randint(1,4)
        print("State: ", str(state))
        stage, direction, reward, state = stage1(grid, direction, crop, state)

    elif stage == 2: #harvesting
        stage = stage2()

    elif stage == 3:
        stage = stage3(observations)

    elif stage == 4: #counting reward
        stage, reward = stage4(observations)      

print()
print("Mission ended")
# Mission has ended.
