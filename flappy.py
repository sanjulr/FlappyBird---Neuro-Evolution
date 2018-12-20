from itertools import cycle
import random
import sys
import numpy as np

import pygame
from pygame.locals import *
from sklearn.neural_network import MLPClassifier
from keras import Sequential
from keras.layers import Dense, LSTM, RNN
from keras.models import clone_model
import time
import warnings

warnings.filterwarnings("ignore")

FPS = 300
SCREENWIDTH = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE = 120  # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)

xrange = range

# Evolution Parameters
best_paravaigal_in_each_gen = []
population_count = 30
paravaigal = []
max_gen = 100

nn_implementation = 'keras'


# nn_implementation = 'sklearn'


class Paravai:

    def __init__(self, no_brain=False):
        self.movementInfo = globals()['movementInfo']
        self.score = self.playerIndex = self.loopIter = self.time_travelled = self.fitness_score = 0

        # player velocity, max velocity, downward accleration, accleration on flap
        self.playerVelY = -9  # player's velocity along Y, default same as playerFlapped
        self.playerMaxVelY = 10  # max vel along Y, max descend speed
        self.playerMinVelY = -8  # min vel along Y, max ascend speed
        self.playerAccY = 1  # players downward accleration
        self.playerRot = 45  # player's rotation
        self.playerVelRot = 3  # angular speed
        self.playerRotThr = 20  # rotation threshold
        self.playerFlapAcc = -9  # players speed on flapping
        self.playerFlapped = False  # True when player flaps

        self.playerIndexGen = self.movementInfo['playerIndexGen']
        self.playerx, self.playery = int(SCREENWIDTH * 0.2), int(self.movementInfo['playery'])
        empty_inputs = [0] * 22

        nn = None

        if not no_brain:
            initial_input = [self.playerx, self.playery, self.playerVelY, self.playerAccY]
            initial_input.extend(empty_inputs)

            # MLPClassifier NN
            if nn_implementation == 'sklearn':
                nn = MLPClassifier(hidden_layer_sizes=(len(initial_input), 24, 18), shuffle=True)
                nn.fit(np.asarray(initial_input).reshape(1, -1), [True])

            # Keras NN
            elif nn_implementation == 'keras':
                nn = Sequential()
                nn.add(Dense(len(initial_input), input_dim=len(initial_input), activation='relu'))
                nn.add(Dense(24, activation='relu'))
                nn.add(Dense(18, activation='relu'))
                nn.add(Dense(1, activation='sigmoid'))
                nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                nn.fit(np.asarray(initial_input).reshape(1, -1), [True], verbose=0, shuffle=True)

        self.brain = nn

    def flap(self):
        if self.playery > -2 * IMAGES['player'][0].get_height():
            self.playerVelY = self.playerFlapAcc
            self.playerFlapped = True
            SOUNDS['wing'].play()

    def yosi(self, upperPipes, lowerPipes):
        upper1_x = upperPipes[0]['x']
        upper2_x = upperPipes[1]['x']
        upper1_y = upperPipes[0]['y']
        upper2_y = upperPipes[1]['y']
        lower1_x = lowerPipes[0]['x']
        lower2_x = lowerPipes[1]['x']
        lower1_y = lowerPipes[0]['y']
        lower2_y = lowerPipes[1]['y']
        upper1_height = upperPipes[0]['height']
        upper2_height = upperPipes[1]['height']
        lower1_height = lowerPipes[0]['height']
        lower2_height = lowerPipes[1]['height']
        upper1_x_distance = upperPipes[0]['x'] - self.playerx
        upper2_x_distance = upperPipes[1]['x'] - self.playerx
        upper1_y_distance = upperPipes[0]['y'] - self.playery
        upper2_y_distance = upperPipes[1]['y'] - self.playery
        lower1_x_distance = lowerPipes[0]['x'] - self.playerx
        lower2_x_distance = lowerPipes[1]['x'] - self.playerx
        lower1_y_distance = lowerPipes[0]['y'] - self.playery
        lower2_y_distance = lowerPipes[1]['y'] - self.playery
        upper_pipe_gap = upper2_x - upper1_x
        lower_pipe_gap = lower2_x - lower1_x
        input_data = np.array([self.playerx, self.playery, self.playerVelY,
                               self.playerAccY, upper1_x, upper2_x, upper1_y, upper2_y, lower1_x,
                               lower2_x, lower1_y, lower2_y, upper1_height, upper2_height,
                               lower1_height, lower2_height, upper1_x_distance, upper2_x_distance,
                               upper1_y_distance, upper2_y_distance, lower1_x_distance,
                               lower2_x_distance,
                               lower1_y_distance, lower2_y_distance, upper_pipe_gap,
                               lower_pipe_gap]).reshape(1,
                                                        -1)
        prediction = 0
        if nn_implementation == 'keras':
            prediction = self.brain.predict(input_data)[0]
        elif nn_implementation == 'sklearn':
            prediction = self.brain.predict_proba(input_data)[0]
        return prediction


def get_fittest(paravaigal, use_one_parent_from_prev_gen=False, use_score=True):
    sum_time_travelled = 0
    for paravai in paravaigal:
        sum_time_travelled += paravai.time_travelled
    for paravai in paravaigal:
        paravai.fitness_score = (paravai.time_travelled / sum_time_travelled) + (
                paravai.score * 100) if use_score else 0
    paravaigal.sort(key=lambda paravai: paravai.fitness_score, reverse=True)
    best_paravaigal = paravaigal[:2]
    if use_one_parent_from_prev_gen and len(best_paravaigal_in_each_gen) > 0:
        for old_paravaigal in reversed(best_paravaigal_in_each_gen):
            for old_paravai in old_paravaigal:
                if old_paravai.fitness_score > best_paravaigal[1].fitness_score:
                    best_paravaigal[1] = old_paravai
    return best_paravaigal


def crossover(weights, rate=0.3, use_grandparents_gene=False):
    new_weights1 = weights[0]
    new_weights2 = weights[1]
    for i in range(int(len(new_weights1) * rate)):
        for j in range(int(len(new_weights2) * rate)):
            new_weights1[i][j] = weights[0][i][j]
            new_weights2[i][j] = weights[1][i][j]
    if use_grandparents_gene and len(weights) >= 4:
        for l in range(int(len(new_weights1) - 1)):
            for m in range(int(len(new_weights2) - 1)):
                if random.uniform(0, 1) > 0.85:
                    new_weights1[l][m] = weights[2][l][m]
                    new_weights2[l][m] = weights[3][l][m]
    return new_weights1, new_weights2


def mutate(weights, rate=0.1, chance=0.25):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if random.uniform(0, 1) > (1 - chance):
                mutation = random.uniform(-rate, rate)
                weights[i][j] += mutation
    return weights


def create_next_population(parents, single_parent=False):
    appa: Paravai = parents[0]
    amma: Paravai = parents[1]
    weights1 = weights2 = weights3 = weights4 = None
    if nn_implementation == 'keras':
        weights1 = appa.brain.get_weights()
        weights2 = amma.brain.get_weights()
    elif nn_implementation == 'sklearn':
        weights1 = appa.brain.coefs_
        weights2 = amma.brain.coefs_
    all_weights = [weights1, weights2]

    if len(best_paravaigal_in_each_gen) > 0:
        grandparents = best_paravaigal_in_each_gen[-1]
        thatha: Paravai = grandparents[0]
        paati: Paravai = grandparents[1]
        if nn_implementation == 'keras':
            weights3 = thatha.brain.get_weights()
            weights4 = paati.brain.get_weights()
        elif nn_implementation == 'sklearn':
            weights3 = thatha.brain.coefs_
            weights4 = paati.brain.coefs_
        all_weights.append(weights3)
        all_weights.append(weights4)

    if single_parent:
        new_weights = [weights1, weights1]
    else:
        new_weights = crossover(all_weights, use_grandparents_gene=False)
    new_weights1 = mutate(new_weights[0], 0.5, 0.25)
    new_weights2 = mutate(new_weights[1], 0.5, 0.25)
    weights = [new_weights1, new_weights2]
    for i in range(population_count):
        kutty = Paravai()
        select = random.randint(0, 1)
        if nn_implementation == 'keras':
            kutty.brain.set_weights(weights[select])
        else:
            kutty.brain.coefs_ = weights[select]
        paravaigal.append(kutty)


def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die'] = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit'] = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point'] = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing'] = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    while True:
        # select random background sprites
        randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

        # select random player sprites
        randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
        )

        # select random pipe sprites
        pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
        )

        # hismask for pipes
        HITMASKS['pipe'] = (
            getHitmask(IMAGES['pipe'][0]),
            getHitmask(IMAGES['pipe'][1]),
        )

        # hitmask for player
        HITMASKS['player'] = (
            getHitmask(IMAGES['player'][0]),
            getHitmask(IMAGES['player'][1]),
            getHitmask(IMAGES['player'][2]),
        )

        globals()['movementInfo'] = showWelcomeAnimation()
        gen_score = 0
        for gen in range(max_gen):
            print("Generation {}".format(gen + 1))
            for i in range(population_count):
                if len(paravaigal) < population_count:
                    paravai = Paravai()
                else:
                    paravai = paravaigal[i]
                crashInfo = mainGame(paravai)
                paravai.time_travelled = crashInfo['time_travelled']
                gen_score += paravai.time_travelled
                paravaigal.append(paravai)
            print("Gen score = {}".format(gen_score))
            gen_score = 0
            fittest_paravaigal = get_fittest(paravaigal, use_one_parent_from_prev_gen=False, use_score=False)
            best_paravaigal_in_each_gen.append(fittest_paravaigal)
            paravaigal.clear()
            create_next_population(fittest_paravaigal, single_parent=False)
            # showGameOverScreen(crashInfo)


def showWelcomeAnimation():
    """Shows welcome screen animation of flappy bird"""
    # index of player to blit on screen
    playerIndex = 0
    playerIndexGen = cycle([0, 1, 2, 1])
    # iterator used to change playerIndex after every 5th iteration
    loopIter = 0

    playerx = int(SCREENWIDTH * 0.2)
    playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

    messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
    messagey = int(SCREENHEIGHT * 0.12)

    basex = 0
    # amount by which base can maximum shift to left
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # player shm for up-down motion on welcome screen
    playerShmVals = {'val': 0, 'dir': 1}

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            # make first flap sound and return values for mainGame
            SOUNDS['wing'].play()
            return {
                'playery': playery + playerShmVals['val'],
                'basex': basex,
                'playerIndexGen': playerIndexGen,
            }

        # adjust playery, playerIndex, basex
        if (loopIter + 1) % 5 == 0:
            playerIndex = next(playerIndexGen)
        loopIter = (loopIter + 1) % 30
        basex = -((-basex + 4) % baseShift)
        playerShm(playerShmVals)

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0, 0))
        SCREEN.blit(IMAGES['player'][playerIndex],
                    (playerx, playery + playerShmVals['val']))
        SCREEN.blit(IMAGES['message'], (messagex, messagey))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))

        pygame.display.update()
        FPSCLOCK.tick(FPS)


def mainGame(paravai):
    time_travelled = 0
    basex = 0
    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    # get 2 new pipes to add to upperPipes lowerPipes list
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()

    # list of upper pipes
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y'], 'height': newPipe1[0]['height']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y'], 'height': newPipe2[0]['height']},
    ]

    # list of lowerpipe
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y'], 'height': newPipe1[1]['height']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y'], 'height': newPipe2[1]['height']},
    ]

    pipeVelX = -4
    while True:
        render_game(basex, baseShift, upperPipes, lowerPipes, pipeVelX)
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                paravai.flap()

        # check for crash here
        crashTest = checkCrash({'x': paravai.playerx, 'y': paravai.playery, 'index': paravai.playerIndex},
                               upperPipes, lowerPipes)
        decision = paravai.yosi(upperPipes, lowerPipes)
        if decision[0] > 0.5:
            paravai.flap()
        if crashTest[0]:
            return {
                'y': paravai.playery,
                'groundCrash': crashTest[1],
                'basex': basex,
                'upperPipes': upperPipes,
                'lowerPipes': lowerPipes,
                'score': paravai.score,
                'playerVelY': paravai.playerVelY,
                'playerRot': paravai.playerRot,
                'time_travelled': time_travelled,
                'brain': paravai.brain
            }

        # check for score
        playerMidPos = paravai.playerx + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                paravai.score += 1
                SOUNDS['point'].play()

        # playerIndex basex change
        if (paravai.loopIter + 1) % 3 == 0:
            paravai.playerIndex = next(paravai.playerIndexGen)
        paravai.loopIter = (paravai.loopIter + 1) % 30

        # rotate the player
        if paravai.playerRot > -90:
            paravai.playerRot -= paravai.playerVelRot

        # player's movement
        if paravai.playerVelY < paravai.playerMaxVelY and not paravai.playerFlapped:
            paravai.playerVelY += paravai.playerAccY
        if paravai.playerFlapped:
            paravai.playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            paravai.playerRot = 45

        playerHeight = IMAGES['player'][paravai.playerIndex].get_height()
        paravai.playery += min(paravai.playerVelY, BASEY - paravai.playery - playerHeight)

        # print score so player overlaps the score
        showScore(paravai.score)

        # Player rotation has a threshold
        visibleRot = paravai.playerRotThr
        if paravai.playerRot <= paravai.playerRotThr:
            visibleRot = paravai.playerRot

        playerSurface = pygame.transform.rotate(IMAGES['player'][paravai.playerIndex], visibleRot)
        SCREEN.blit(playerSurface, (paravai.playerx, paravai.playery))

        time_travelled += 1
        pygame.display.update()
        FPSCLOCK.tick(FPS)


def render_game(basex, baseShift, upperPipes, lowerPipes, pipeVelX):
    basex = -((-basex + 100) % baseShift)

    # move pipes to left
    for uPipe, lPipe in zip(upperPipes, lowerPipes):
        uPipe['x'] += pipeVelX
        lPipe['x'] += pipeVelX

    # add new pipe when first pipe is about to touch left of screen
    if 0 < upperPipes[0]['x'] < 5:
        newPipe = getRandomPipe()
        upperPipes.append(newPipe[0])
        lowerPipes.append(newPipe[1])

    # remove first pipe if its out of the screen
    if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
        upperPipes.pop(0)
        lowerPipes.pop(0)

    # draw sprites
    SCREEN.blit(IMAGES['background'], (0, 0))

    for uPipe, lPipe in zip(upperPipes, lowerPipes):
        SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
        SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

    SCREEN.blit(IMAGES['base'], (basex, BASEY))


def showGameOverScreen(crashInfo):
    """crashes the player down ans shows gameover image"""
    score = crashInfo['score']
    playerx = SCREENWIDTH * 0.2
    playery = crashInfo['y']
    playerHeight = IMAGES['player'][0].get_height()
    playerVelY = crashInfo['playerVelY']
    playerAccY = 2
    playerRot = crashInfo['playerRot']
    playerVelRot = 7

    basex = crashInfo['basex']

    upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

    # play hit and die sounds
    SOUNDS['hit'].play()
    if not crashInfo['groundCrash']:
        SOUNDS['die'].play()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if playery + playerHeight >= BASEY - 1:
                    return

        # player y shift
        if playery + playerHeight < BASEY - 1:
            playery += min(playerVelY, BASEY - playery - playerHeight)

        # player velocity change
        if playerVelY < 15:
            playerVelY += playerAccY

        # rotate only when it's a pipe crash
        if not crashInfo['groundCrash']:
            if playerRot > -90:
                playerRot -= playerVelRot

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (basex, BASEY))
        showScore(score)

        playerSurface = pygame.transform.rotate(IMAGES['player'][1], playerRot)
        SCREEN.blit(playerSurface, (playerx, playery))

        FPSCLOCK.tick(FPS)
        pygame.display.update()


def playerShm(playerShm):
    """oscillates the value of playerShm['val'] between 8 and -8"""
    if abs(playerShm['val']) == 8:
        playerShm['dir'] *= -1

    if playerShm['dir'] == 1:
        playerShm['val'] += 1
    else:
        playerShm['val'] -= 1


def getRandomPipe():
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10
    pipeY_upper = gapY - pipeHeight
    pipeY_lower = gapY + PIPEGAPSIZE
    upper_height = pipeHeight + pipeY_upper
    lower_height = pipeHeight - pipeY_lower
    return [
        {'x': pipeX, 'y': pipeY_upper, 'height': pipeHeight - pipeY_upper},  # upper pipe
        {'x': pipeX, 'y': pipeY_lower, 'height': pipeHeight - pipeY_lower},  # lower pipe
    ]


def showScore(score):
    """displays score in center of screen"""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = 0  # total width of all numbers to be printed

    for digit in scoreDigits:
        totalWidth += IMAGES['numbers'][digit].get_width()

    Xoffset = (SCREENWIDTH - totalWidth) / 2

    for digit in scoreDigits:
        SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """returns True if player collders with base or pipes."""
    pi = player['index']
    player['w'] = IMAGES['player'][0].get_width()
    player['h'] = IMAGES['player'][0].get_height()

    # if player crashes into ground or above screen
    if (player['y'] + player['h'] >= BASEY - 1) or (player['y'] + player['h'] < 0):
        return [True, True]
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                                 player['w'], player['h'])
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return [True, False]

    return [False, False]


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in xrange(rect.width):
        for y in xrange(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False


def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


if __name__ == '__main__':
    main()
