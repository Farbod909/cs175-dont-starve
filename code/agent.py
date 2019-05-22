from __future__ import print_function
from builtins import range
import MalmoPython
import os
import sys
import time
import json
import random
import conv_net as cnn
from replay import ReplayMemory

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
                        <InventoryItem slot="8" type="stick" quantity="1"/>
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

class Agent(object):
    def __init__(self):
        self.started = False
        self.finished = False
        self.direction = "east"
        self.grid = [''] * 25
        self.state = [0] * farm_size**2
        self.state[0] = -1
        self.timeAlive = 0

    def nextTick(self):
        print(".", end="")
        for x in range(2): #I'm pretty sure the grid views only get updated every other tick
            world_state = agent_host.getWorldState()
            while world_state.number_of_observations_since_last_state == 0:
                time.sleep(.01)
                world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)
            
        msg = world_state.observations[-1].text
        self.observations = json.loads(msg)
        self.grid = self.observations.get(u'croplocal', 0)

        #because farmland is not a full block, the grid observations sometimes include the ground, so we remove it
        if self.grid[0] == "stone" or self.grid[0] == "dirt" or self.grid[0] == "farmland" or self.grid[0] == "water":
            del self.grid[0:25]

    def updateDirection(self):
        if self.direction == "east" and self.grid[14] == "birch_fence":
            if self.grid[17] == "birch_fence":
                self.finished = True
            self.direction = "south"
        elif self.direction == "west" and self.grid[10] == "birch_fence":
            if self.grid[17] == "birch_fence":
                self.finished = True
            self.direction = "south"
        elif self.direction == "south":
            if self.grid[11] == "birch_fence":
                self.direction = "east"
            else:
                self.direction = "west"

    def updateState(self, crop):
        pos = self.state.index(-1)
        self.state[pos] = crop
        if self.direction == "east":
            self.state[pos+1] = -1
        elif self.direction == "west":
            self.state[pos-1] = -1
        elif self.direction == "south" and len(self.state) > pos + farm_size:
            self.state[pos+farm_size] = -1

    def setup(self):
        if self.grid[7] != "birch_fence":
            agent_host.sendCommand("movenorth 1")
        elif self.grid[11] != "birch_fence":
            agent_host.sendCommand("movewest 1")
        else:
            self.started = True

    def cleanup(self):
        self.nextTick()
        #select empty hotbar slot because sending chat commands sometimes causes the agent to right click
        agent_host.sendCommand("hotbar.9 1")
        agent_host.sendCommand("hotbar.9 0")
        #grow crops
        agent_host.sendCommand("chat /tp 0 ~ 0")
        time.sleep(.5)
        agent_host.sendCommand("chat /gamerule randomTickSpeed 10000")
        time.sleep(1)
        agent_host.sendCommand("chat /gamerule randomTickSpeed 1")
        #harvest crops
        self.nextTick()
        full_grid = self.observations.get(u'cropfull', 0)
        print("Full grid: " + str(full_grid))
        agent_host.sendCommand('chat /fill -' + str(farm_radius) + ' 227 -' + str(farm_radius) + ' ' + str(farm_radius) + ' 227 ' + str(farm_radius) + ' air 0 destroy') #this needs to vary if the size of the field changes
        time.sleep(2)
        agent_host.sendCommand("chat /tp @e[type=item] @p")
        time.sleep(2)
        #count crops
        self.nextTick()
        wheat, carrot, potato, beetroot = 0, 0, 0, 0
        for i in range(0,41):
            item = self.observations.get(u'InventorySlot_'+str(i)+'_item', 0)
            if item == "wheat":
                wheat += self.observations.get(u'InventorySlot_'+str(i)+'_size', 0)
            elif item == "carrot":
                carrot += self.observations.get(u'InventorySlot_'+str(i)+'_size', 0)
            elif item == "potato":
                potato += self.observations.get(u'InventorySlot_'+str(i)+'_size', 0)
            elif item == "beetroot":
                beetroot += self.observations.get(u'InventorySlot_'+str(i)+'_size', 0)
        #if initial resources change, the 64's below need to change as well
        print("\nHarvested " + str(wheat) + " wheat, " + str(carrot-64) + " carrots, " + str(potato-64) + " potatoes, " + str(beetroot) + " beetroots.")

        #subtract the initial carrots and potatoes, and account for harvesting multiples at random from grown crops
        reward = wheat + (carrot - 64) / 1.71 + (potato - 64) / 1.71 + beetroot
        print("Reward is "+str(reward) + " out of " + str(farm_size**2)) #can technically be higher with good RNG
        return reward

    def run(self, crop):
        while not self.started:
            self.nextTick()
            self.setup()

        self.nextTick()
        if not self.state[0] == -1: #skip the first move to plant in the top left corner
            agent_host.sendCommand("move"+self.direction+" 1")
        self.updateDirection()
        
        agent_host.sendCommand("hotbar." + str(crop) + " 1")
        agent_host.sendCommand("hotbar." + str(crop) + " 0")
        agent_host.sendCommand("use 1")
        self.updateState(crop)

        if self.finished:
            return self.cleanup()

        return 0



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

    agent = Agent()

    # Loop until mission ends:
    while not agent.finished:
        crop = random.randint(1,4)
        last_state = agent.state.copy()
        reward = agent.run(crop)
        memory.push(last_state, crop, agent.state.copy(), reward)
            
    print(agent.state)   
    memory.print_replay()
    #net.forward(agent.state)

    print()
    print("Mission ended")
    agent_host.sendCommand("chat /kill @p")
    # Mission has ended.
