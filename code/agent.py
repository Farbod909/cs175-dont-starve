from __future__ import print_function
from builtins import range
import time
import json
import random


class Agent(object):
    def __init__(self, agent_host):
        self.agent_host = agent_host
        self.nextTick()
        self.farm_size = int(len(self.observations.get(u'cropfull', 0)) ** 0.5)
        self.started = False
        self.finished = False
        self.direction = "north"
        self.xpos = 0
        self.zpos = 0
        self.state = [0] * self.farm_size**2
        self.state[0] = -1

    def nextTick(self):
        print(".", end="")
        world_state = self.agent_host.getWorldState()
        while world_state.number_of_observations_since_last_state == 0:
            time.sleep(.01)
            world_state = self.agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
            
        msg = world_state.observations[-1].text
        self.observations = json.loads(msg)
        #self.grid = self.observations.get(u'croplocal', 0)

        #because farmland is not a full block, the grid observations sometimes include the ground, so we remove it
        #if self.grid[0] == "stone" or self.grid[0] == "dirt" or self.grid[0] == "farmland" or self.grid[0] == "water":
        #    del self.grid[0:25]

    def updateDirection(self):
        if self.direction == "east" and self.xpos == int((self.farm_size-1)/2):
            if self.zpos == int((self.farm_size-1)/2):
                self.finished = True
            self.direction = "south"
        elif self.direction == "west" and self.xpos == int((self.farm_size-1)/-2):
            if self.zpos == int((self.farm_size-1)/2):
                self.finished = True
            self.direction = "south"
        elif self.direction == "south":
            if self.xpos == int((self.farm_size-1)/-2):
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
        elif self.direction == "south" and len(self.state) > pos + self.farm_size:
            self.state[pos+self.farm_size] = -1

    def move(self):
        self.agent_host.sendCommand("move"+self.direction+" 1")
        if self.direction == "north":
            self.zpos -= 1
        elif self.direction == "south":
            self.zpos += 1
        elif self.direction == "west":
            self.xpos -= 1
        elif self.direction == "east":
            self.xpos += 1
        print(str(self.xpos) + " " + str(self.zpos))

    def setup(self): #move to northwest corner
        while self.zpos > (self.farm_size-1)/-2:
            self.nextTick()
            self.move()
        self.direction = "west"
        while self.xpos > (self.farm_size-1)/-2:
            self.nextTick()
            self.move()
        self.direction = "east"
        self.started = True

    def cleanup(self):
        self.nextTick()
        #select seeds because sending chat commands sometimes causes the agent to right click
        self.agent_host.sendCommand("hotbar.1 1")
        self.agent_host.sendCommand("hotbar.1 0")
        #grow crops
        self.agent_host.sendCommand("chat /tp 0 ~ 0")
        time.sleep(.1)
        self.agent_host.sendCommand("chat /gamerule randomTickSpeed 10000")
        time.sleep(1)
        self.agent_host.sendCommand("chat /gamerule randomTickSpeed 1")
        #harvest crops
        self.nextTick()
        full_grid = self.observations.get(u'cropfull', 0)
        print("Full grid: " + str(full_grid))
        self.agent_host.sendCommand('chat /fill -' + str((self.farm_size-1)/2) + ' 227 -' + str((self.farm_size-1)/2) + ' ' + str((self.farm_size-1)/2) + ' 227 ' + str((self.farm_size-1)/2) + ' air 0 destroy') #this needs to vary if the size of the field changes
        time.sleep(.5) #increase this if you get lag
        self.nextTick()
        self.agent_host.sendCommand("chat /tp @e[type=item] @p")
        time.sleep(.5) #increase this too
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
        print("Reward is "+str(reward) + " out of " + str(self.farm_size**2)) #can technically be higher with good RNG
        return reward

    def run(self, crop):
        if not self.started:
            self.setup()

        self.nextTick()
        if not self.state[0] == -1: #skip the first move to plant in the top left corner
            self.move()
        self.updateDirection()
        
        self.agent_host.sendCommand("hotbar." + str(crop) + " 1")
        self.agent_host.sendCommand("hotbar." + str(crop) + " 0")
        self.agent_host.sendCommand("use 1")
        self.updateState(crop)

        if self.finished:
            return self.cleanup()

        return 0

