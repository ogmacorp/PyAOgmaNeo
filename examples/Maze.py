import numpy as np
import pygame
import pygame.surfarray
from copy import copy
import pyaogmaneo as neo

# The maze. 1 for walls, 0 for empty space, 2 is spawn, other numbers are possible goal locations
#maze = [
#    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
#    [ 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1 ],
#    [ 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1 ],
#    [ 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 ],
#    [ 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 ],
#    [ 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1 ],
#    [ 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 ],
#    [ 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1 ],
#    [ 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1 ],
#    [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1 ],
#    [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1 ],
#    [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
#]

# The maze. 1 for walls, 0 for empty space
maze = [
    [ 1, 1, 1, 1, 1, 1, 1, 1 ],
    [ 1, 0, 1, 0, 0, 0, 0, 1 ],
    [ 1, 0, 0, 0, 1, 1, 0, 1 ],
    [ 1, 0, 0, 0, 0, 0, 0, 1 ],
    [ 1, 0, 1, 0, 0, 0, 0, 1 ],
    [ 1, 0, 1, 1, 0, 1, 0, 1 ],
    [ 1, 0, 0, 0, 0, 1, 0, 1 ],
    [ 1, 1, 1, 1, 1, 1, 1, 1 ]
]

mazeSize = ( len(maze), len(maze[0]) )

########################## Window ###########################

screen = pygame.display.set_mode((800, 800))

# Pre-render maze image
mazeRGB = np.zeros((mazeSize[1], mazeSize[0], 3), dtype=np.uint8)

for y in range(len(maze[1])):
    for x in range(len(maze[0])):
        if maze[x][y] == 1:
            mazeRGB[y, x, :] = np.array([ 255, 255, 255 ], dtype=np.uint8)

# Scale maze surface to size of screen
mazeSurf = pygame.transform.scale(pygame.surfarray.make_surface(mazeRGB), screen.get_size())

# Ratio of screen to maze size (tile size)
sizeRatio = (screen.get_size()[0] / mazeSize[0], screen.get_size()[1] / mazeSize[1])

########################### Agent ###########################

# Vision configuration
visRadius = 3 # Radius of visual field
visDiam = visRadius * 2 + 1
visArea = visDiam * visDiam
numActions = 5 # Cardinal directions + standing still
waitSteps = 32 # Number of steps to wait at goal before getting top hidden state
subSteps = 40 # Number of steps in train mode per frame

# Number of threads to use
neo.setNumThreads(4)

# Hierarchy
lds = []

for i in range(3):
    ld = neo.LayerDesc()

    ld.hiddenSize = (4, 4, 16)

    lds.append(ld)

h = neo.Hierarchy()
h.initRandom([ neo.IODesc((visDiam, visDiam, 2), neo.none), neo.IODesc((1, 1, 5), neo.prediction) ], lds)

# Set action importance low otherwise will favor just sticking to the goal action too much
h.setInputImportance(1, 0.0)

# Goal is based on top-most CSDR
goal = h.getTopHiddenCIs()

######################### Game Loop ########################

# Get observation
def getVisual(pos):
    # Determine vision - rectangle around agent
    v = visArea * [ 0 ]

    for dx in range(-visRadius, visRadius + 1):
        for dy in range(-visRadius, visRadius + 1):
            vPos = [ pos[0] + dx, pos[1] + dy ]

            if vPos[0] >= 0 and vPos[1] >= 0 and vPos[0] < mazeSize[0] and vPos[1] < mazeSize[1]:
                v[dx + visRadius + (dy + visRadius) * visDiam] = maze[vPos[1]][vPos[0]]

    return v

# For applying actions
def getNewPos(agentPos, action):
    pos = copy(agentPos)

    if action == 1:
        pos[0] += 1
    elif action == 2:
        pos[1] += 1
    elif action == 3:
        pos[0] -= 1
    elif action == 4:
        pos[1] -= 1

    if maze[pos[1]][pos[0]] == 1:
        pos = agentPos

    return pos

agentPos = [1, 1] # Set to spawn position
action = 0
trainMode = False # Flag for training (sped up) mode
stepTicks = 0 # Timer
numTicksPerStep = 20

keysPressedPrev = pygame.key.get_pressed()
mouseButtonsPrev = pygame.mouse.get_pressed()

running = True

start_time = pygame.time.get_ticks()

while running:
    end_time = pygame.time.get_ticks()
    dt = end_time - start_time
    start_time = end_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keysPressed = pygame.key.get_pressed()
    mouseButtons = pygame.mouse.get_pressed()

    if keysPressed[pygame.K_q]:
        running = False

    if keysPressed[pygame.K_t] and not keysPressedPrev[pygame.K_t]:
        trainMode = not trainMode # Toggle

    if mouseButtons[0] and not mouseButtonsPrev[0]:
        # Set goal by teleporting agent to goal position, having it sit there a bit and recording its current top hidden state
        oldPos = copy(agentPos)

        mousePos = pygame.mouse.get_pos()

        agentPos = (min(mazeSize[0] - 1, max(0, int(mousePos[0] / sizeRatio[0]))), min(mazeSize[1] - 1, max(0, int(mousePos[1] / sizeRatio[1]))))

        for s in range(waitSteps):
            v = getVisual(agentPos)

            h.step([ v, [ 0 ] ], h.getTopHiddenCIs(), False) # 0 is standing still action, goal is set to top hidden state as it isn't used yet anyways

        # Record goal CSDR
        goal = h.getTopHiddenCIs()

        # Revert
        agentPos = oldPos

        print("Goal set: " + str(goal))

    if trainMode:
        stepTicks = 0

        for s in range(subSteps): # If in train mode, perform many steps
            v = getVisual(agentPos)

            h.step([ v, [ action ] ], h.getTopHiddenCIs(), True) # Goal just set to current top hidden state as it isn't used yet anyways

            action = np.random.randint(0, numActions) # Explore randomly

            agentPos = getNewPos(agentPos, action) # Move

    elif stepTicks >= numTicksPerStep: # Step agent if takeStep (to slow down the game for viewing)
        stepTicks = 0

        v = getVisual(agentPos)

        h.step([ v, [ action ] ], goal, False) # Goal set and learning disabled

        action = h.getPredictionCIs(1)[0] # Get the only action in the action IO layer

        agentPos = getNewPos(agentPos, action) # Move

        #print("Hidden state: " + str(h.getTopHiddenCIs()))

    # Draw maze
    screen.blit(mazeSurf, (0, 0))

    # Draw agent
    pygame.draw.circle(screen, (0, 255, 0), (agentPos[0] * sizeRatio[0] + sizeRatio[0] * 0.5, agentPos[1] * sizeRatio[1] + sizeRatio[1] * 0.5), sizeRatio[0] * 0.5)

    # Draw vision rectangle
    pygame.draw.rect(screen, (255, 0, 0), ((agentPos[0] - visRadius) * sizeRatio[0], (agentPos[1] - visRadius) * sizeRatio[1], visDiam * sizeRatio[0], visDiam * sizeRatio[1]), 1, border_radius=2)

    pygame.display.flip()

    keysPressedPrev = keysPressed
    mouseButtonsPrev = mouseButtons

    stepTicks += 1

    pygame.time.delay(max(0, 1000 // 60 - dt))
