import time
import random
import shelve
import numpy as np
import pdb
from importlib import reload
import matplotlib.pyplot as plt
import cellular

reload(cellular)
# import qlearn_mod_random as qlearn # to use the alternative exploration method

import qlearn  # to use standard exploration method

reload(qlearn)
initial_state_value=[]
directions = 5

lookdist = 2
lookcells = []

for i in range(-lookdist, lookdist + 1):
    for j in range(-lookdist, lookdist + 1):
        if (abs(i) + abs(j) <= lookdist) and (i != 0 or j != 0) and (abs(i)!=abs(j)):
            lookcells.append((i, j))

lookcells.append((i,j))
def pickRandomLocation():
    while 1:
        x = 1
        y = 1
        cell = world.getCell(x, y)
        if not (cell.wall or len(cell.agents) > 0):
            return cell


class Cell(cellular.Cell):
    wall = False

    def colour(self):
        if self.wall:
            return 'black'
        else:
            return 'white'

    def load(self, data):
        if data == 'X':
            self.wall = True
        else:
            self.wall = False


class Police(cellular.Agent):
    cell = None
    score = 0
    colour = 'orange'

    def move(self):
        move = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random_move = random.choice(move)
        # check if it is in maze
        return random_move

    def update(self):
        cell = self.cell

        if cell != robber.cell:
            value = self.move()
            self.goTowards(world.getCell(value[0], value[1]))
            while cell == self.cell:
                self.goInDirection(random.randrange(4))


class Bank(cellular.Agent):
    colour = 'yellow'

    def update(self):
        pass


class Robber(cellular.Agent):
    colour = 'gray'

    def __init__(self):
        self.ai = None
        self.ai = qlearn.QLearn(actions=range(directions),
                                alpha=1, gamma=0.8, epsilon=0.5)
        self.caught = 0
        self.heist = 0
        self.lastState = None
        self.lastAction = None
        self.reward = 0
        self.step=0

    def update(self):
        state = self.calcState()
        reward = -1
        self.reward += reward
        if self.cell == police.cell:
            self.caught += 1
            reward = -10
            self.reward += reward
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, reward, state)
            #self.lastState = None
            self.cell = world.getCell(1, 1)
            police.cell = world.getCell(4, 4)
            # print(self.reward)
            return

        if self.cell == bank.cell:
            self.heist += 1
            reward = 1
            self.reward += reward
            #bank.cell = world.getCell(2, 2)
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, reward, state)

        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, reward, state)

        initial_state = (1, 1, 1, 1, 0, 0, 0, 0, 0)
        val = []
        for i in robber.ai.q:
            if i[0] == initial_state:
                # get all value for this state
                val.append(robber.ai.q.get(i))
        if (len(val) != 0):
            initial_state_value.append(max(val))
        state = self.calcState()
        action = self.ai.chooseAction(state)
        self.lastState = state
        self.lastAction = action

        self.goInDirection(action)

    def calcState(self):
        def cellvalue(cell):
            if police.cell is not None and (cell.x == police.cell.x and
                                            cell.y == police.cell.y):
                return 3
            elif bank.cell is not None and (cell.x == bank.cell.x and
                                            cell.y == bank.cell.y):
                return 2
            else:
                return 1 if cell.wall else 0

        return tuple([cellvalue(self.world.getWrappedCell(self.cell.x + j, self.cell.y + i))
                      for i, j in lookcells])


robber = Robber()
police = Police()
bank = Bank()
world = cellular.World(Cell, directions=directions, filename='waco.txt')
world.age = 0

world.addAgent(bank, x=2, y=2)
world.addAgent(police, x=4, y=4)
world.addAgent(robber, x=1, y=1)
endAge = world.age + 10050000

while world.age < endAge:
    world.update()

    # if world.age % 100 == 0:
    #     mouse.ai.epsilon = (epsilony[0] if world.age < epsilonx[0] else
    #                         epsilony[1] if world.age > epsilonx[1] else
    #                         epsilonm*(world.age - epsilonx[0]) + epsilony[0])

    robber.ai.alpha = 1 / (world.age ** (2 / 3))
    if world.age % 1000 == 0:
        print("{:d}, e: {:0.2f}, W: {:d}, L: {:d}, R: {:d}" \
               .format(world.age, robber.ai.epsilon, robber.heist, robber.caught, robber.reward))

        robber.heist = 0
        robber.caught = 0
        robber.reward = 0


tit='epsilon 0.5'
plt.plot(range(len(initial_state_value)), initial_state_value, label='Initial state')
plt.xlabel('time')
plt.ylabel('Value function')
plt.title(tit)
plt.legend()
plt.show()

# world.display.activate(size=30)
# world.display.delay = 1
#
# while 1:
#     world.update()
