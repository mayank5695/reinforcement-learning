import time
import random
import shelve

import pdb
from importlib import reload
import sarsa
import cellular
import matplotlib.pyplot as plt
import seaborn as sns
reload(cellular)
reload(sarsa)

directions = 5

lookdist = 2
lookcells = []
reward_episode = []

initial_state_value=[]
for i in range(-lookdist, lookdist + 1):
    for j in range(-lookdist, lookdist + 1):
        if (abs(i) + abs(j) <= lookdist) and (i != 0 or j != 0) and (abs(i) != abs(j)):
            lookcells.append((i, j))
lookcells.append((0, 0))


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
        self.ai = sarsa.Sarsa(actions=range(directions),
                              alpha=1, gamma=0.8, epsilon=0.1)
        self.caught = 0
        self.heist = 0
        self.lastState = None
        self.lastAction = None
        self.reward = 0
        self.step = 0

    def update(self):
        state = self.calcState()

        reward = -1
        self.reward += (self.ai.gamma ** self.step * reward)
        action = self.ai.chooseAction(state)

        if self.cell == police.cell:
            self.caught += 1
            reward = -10
            self.reward += (self.ai.gamma ** self.step * reward)
            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, reward, state, action)
            #self.lastState = None

            self.cell = world.getCell(1, 1)
            police.cell = world.getCell(4, 4)

            # reward_episode.append(self.reward)
            # self.reward = 0
            # self.step=0
            return

        if self.cell == bank.cell:
            self.heist += 1
            reward = 1
            self.reward += (self.ai.gamma ** self.step * reward)
            # bank.cell = world.getCell(2, 2)

            if self.lastState is not None:
                self.ai.learn(self.lastState, self.lastAction, reward, state, action)

        if self.lastState is not None:
            self.ai.learn(self.lastState, self.lastAction, reward, state, action)

        #print("initial state", state)
        initial_state=(1, 1, 1, 1, 0, 0, 0, 0, 0)

        val=[]
        for i in robber.ai.q:
            if i[0]==initial_state:
                #get all value for this state
                val.append(robber.ai.q.get(i))
        if(len(val)!=0):
            initial_state_value.append(max(val))

        state = self.calcState()
        action = self.ai.chooseAction(state)
        self.lastState = state
        self.lastAction = action

        self.goInDirection(action)
        self.step += 1

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

epsilon=[0.1,0.2,0.3,0.4,0.5,0.6,0.7]

for ep in epsilon:
    initial_state_value=[]
    robber = Robber()
    police = Police()
    bank = Bank()
    world = cellular.World(Cell, directions=directions, filename='waco.txt')
    world.age = 0

    world.addAgent(bank, x=2, y=2)
    world.addAgent(police, x=4, y=4)
    world.addAgent(robber, x=1, y=1)
    epsilonx = (0, 100000)
    epsilony = (0.1, 0)
    epsilonm = (epsilony[1] - epsilony[0]) / (epsilonx[1] - epsilonx[0])

    endAge = world.age + 5000000
    robber.ai.epsilon=ep
    print("plotting for ",ep)

    while world.age < endAge:
        world.update()

        # if world.age % 1000 == 0:
        #     robber.ai.epsilon = (epsilony[0] if world.age < epsilonx[0] else
        #                         epsilony[1] if world.age > epsilonx[1] else
        #                         epsilonm*(world.age - epsilonx[0]) + epsilony[0])

        robber.ai.alpha = 1 / (world.age**(2/3))
        if world.age % 1000 == 0:
            # print("{:d}, e: {:0.2f}, W: {:d}, L: {:d}, R: {:.15f}, A: {:0.9f}" \
            #       .format(world.age, robber.ai.epsilon, robber.heist, robber.caught, robber.reward, robber.ai.alpha))
            a = "{:d},{:0.2f},{:d},{:d},{:0.15f}\n" \
                .format(world.age, robber.ai.epsilon, robber.heist, robber.caught, robber.reward)

            reward_episode.append(robber.reward)
            robber.heist = 0
            robber.caught = 0
            robber.reward = 0
            robber.step=0
            # #print(robber.ai.q)
            # file_sarsa = open(filename, 'a')
            # file_sarsa.write(a)
            # file_sarsa.close()

    tit='epsilon '+str(ep)
    plt.plot(range(len(initial_state_value)), initial_state_value, label=tit)
    plt.xlabel('time')
    plt.ylabel('Value function')
    #plt.title(tit)
    plt.legend()
plt.show()