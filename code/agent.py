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

missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Don't Starve</Summary>
              </About>
              
              <ServerSection>
                <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;7,226*1;21;"/>
                  <DrawingDecorator>
                      <DrawCuboid x1="-9" y1="226" z1="-9" x2="9" y2="226" z2="9" type="grass"/>
                      <DrawCuboid x1="-9" y1="227" z1="-9" x2="9" y2="227" z2="9" type="air"/>
                      <DrawLine x1="-9" y1="227" z1="-9" x2="9" y2="227" z2="-9" type="birch_fence"/>
                      <DrawLine x1="-9" y1="227" z1="-9" x2="-9" y2="227" z2="9" type="birch_fence"/>
                      <DrawLine x1="9" y1="227" z1="9" x2="9" y2="227" z2="-9" type="birch_fence"/>
                      <DrawLine x1="9" y1="227" z1="9" x2="-9" y2="227" z2="9" type="birch_fence"/>
                      <DrawBlock x="4" y="226" z="4" type="water"/>
                      <DrawBlock x="-4" y="226" z="4" type="water"/>
                      <DrawBlock x="4" y="226" z="-4" type="water"/>
                      <DrawBlock x="-4" y="226" z="-4" type="water"/>
                      <DrawBlock x="4" y="227" z="4" type="waterlily"/>
                      <DrawBlock x="-4" y="227" z="4" type="waterlily"/>
                      <DrawBlock x="4" y="227" z="-4" type="waterlily"/>
                      <DrawBlock x="-4" y="227" z="-4" type="waterlily"/>
                  </DrawingDecorator>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>Wilson</Name>
                <AgentStart>
                    <Placement x="0.5" y="227" z="0.5" pitch="90" yaw="180"/>
                    <Inventory>
                        <InventoryItem slot="0" type="diamond_hoe"/>
                        <InventoryItem slot="1" type="wheat_seeds" quantity="64"/>
                        <InventoryItem slot="2" type="carrot" quantity="64"/>
                        <InventoryItem slot="3" type="potato" quantity="64"/>
                        <InventoryItem slot="4" type="beetroot_seeds" quantity="64"/>
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
                      <Grid name="cropfull">
                        <min x="-8" y="1" z="-8"/>
                        <max x="8" y="1" z="8"/>
                      </Grid>
                  </ObservationFromGrid>
                  <ObservationFromFullInventory/>
                  <ChatCommands/>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''

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
direction = "none"
planting = False

# Loop until mission ends:
while stage < 5: #while world_state.is_mission_running:
    print(".", end="")
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    while len(world_state.observations) == 0:
        time.sleep(.1)
        world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)
        
    msg = world_state.observations[-1].text
    observations = json.loads(msg)
    grid = observations.get(u'croplocal', 0)

    #because farmland is not a full block, the grid observations sometimes include the ground, so we remove it
    if grid[0] == "stone" or grid[0] == "grass" or grid[0] == "farmland" or grid[0] == "water":
        del grid[0:25]
       
    if stage == 0: #move to corner and start planting
        if grid[7] == "air":
            agent_host.sendCommand("movenorth 1")
        elif grid[11] == "air":
            agent_host.sendCommand("movewest 1")
        else:
            stage = 1

    if stage == 1: #planting
        if grid[12] == "waterlily":
            planting = False
        
        if not planting: #move and use hoe
            planting = True
            direction = chooseDirection(direction, grid)
            if direction == "nextstage":
                stage = 2
                    
            agent_host.sendCommand("move"+direction+" 1")
            
            if direction == "none":
                direction = "east"

            agent_host.sendCommand("hotbar.1 1")
            agent_host.sendCommand("hotbar.1 0")
            agent_host.sendCommand("use 1") 

        else: #plant seeds
            planting = False
            cropchoice = random.randint(2,5) #ML here to choose the right crop based on grid
            
            agent_host.sendCommand("hotbar." + str(cropchoice) + " 1")
            agent_host.sendCommand("hotbar." + str(cropchoice) + " 0")
            agent_host.sendCommand("use 1")

    if stage == 2: #harvesting
        stage = 3
        agent_host.sendCommand("chat /gamerule randomTickSpeed 10000")
        time.sleep(1)
        agent_host.sendCommand("chat /gamerule randomTickSpeed 1")
        time.sleep(.1)
        agent_host.sendCommand("chat /tp 0 ~ 0")
        time.sleep(.5)

    elif stage == 3:
        stage = 4
        full_grid = observations.get(u'cropfull', 0)
        print("Full grid: " + str(full_grid))
        agent_host.sendCommand("chat /fill -8 227 -8 8 227 8 air 0 destroy") #this needs to vary if the size of the field changes
        time.sleep(10)
        agent_host.sendCommand("chat /tp @e[type=item] @p")
        time.sleep(2)

    elif stage == 4: #counting reward
        stage = 5
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
        print("Reward is "+str(reward) + " out of 256") #can technically be higher with good RNG
        
        

print()
print("Mission ended")
# Mission has ended.
