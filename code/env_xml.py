
def generate_env():
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
    return missionXML
