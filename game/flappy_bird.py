import random
from itertools import cycle

import pygame


def load():
    # path of player with different states
    PLAYER_PATH = (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png'
    )

    # path of background
    BACKGROUND_PATH = 'assets/sprites/background-black.png'

    # path of pipe
    PIPE_PATH = 'assets/sprites/pipe-green.png'

    IMAGES, HITMASKS = {}, {}

    # numbers sprites for score display
    IMAGES['numbers'] = [
        pygame.image.load('assets/sprites/{}.png'.format(idx)).convert_alpha()
        for idx in range(10)
    ]

    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # select random background sprites
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()

    # select random player sprites
    IMAGES['player'] = [
        pygame.image.load(PLAYER_PATH[idx]).convert_alpha()
        for idx in range(3)
    ]

    # select random pipe sprites
    IMAGES['pipe'] = (
        pygame.transform.rotate(
            pygame.image.load(PIPE_PATH).convert_alpha(), 180),
        pygame.image.load(PIPE_PATH).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = [
        getHitmask(IMAGES['pipe'][idx])
        for idx in range(2)
    ]

    # hitmask for player
    HITMASKS['player'] = [
        getHitmask(IMAGES['player'][idx])
        for idx in range(3)
    ]

    return IMAGES, HITMASKS


def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512

pygame.init()
FPSCLOCK = pygame.time.Clock()
SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
pygame.display.set_caption('Flappy Bird')

IMAGES, HITMASKS = load()
PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79

PLAYER_WIDTH = IMAGES['player'][0].get_width()
PLAYER_HEIGHT = IMAGES['player'][0].get_height()
PIPE_WIDTH = IMAGES['pipe'][0].get_width()
PIPE_HEIGHT = IMAGES['pipe'][0].get_height()
BACKGROUND_WIDTH = IMAGES['background'].get_width()

PLAYER_INDEX_GEN = cycle([0, 1, 2, 1])


class GameState:
    def __init__(self):
        self.score = self.playerIndex = self.loopIter = 0
        self.playerx = int(SCREENWIDTH * 0.2)
        self.playery = int((SCREENHEIGHT - PLAYER_HEIGHT) / 2)
        self.basex = 0
        self.baseShift = IMAGES['base'].get_width() - BACKGROUND_WIDTH
        self.difficulty = 1 # -1 = easy, 0 = normal, 1 = difficult

        newPipe1 = getRandomPipe(self.difficulty)
        if self.difficulty == -1:
            self.upperPipes = [
                {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            ]
            self.lowerPipes = [
                {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            ]
        else:
            newPipe2 = getRandomPipe(self.difficulty)
            self.upperPipes = [
                {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
                {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
            ]
            self.lowerPipes = [
                {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
                {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
            ]

        # player velocity, max velocity, downward accleration, accleration on flap
        if self.difficulty == -1:
            #Less drastic movement, falls fast enough to continue to play the game, but don't have to worry about upward
            #  flap running into things as much
            self.pipeVelX = -4  # player velocity, max velocity, downward accleration, accleration on flap
            self.playerVelY = -2  # player's velocity along Y, default same as playerFlapped
            self.playerMaxVelY = 10  # max vel along Y, max descend speed
            self.playerMinVelY = -8  # min vel along Y, max ascend speed
            self.playerAccY = 1  # players downward accleration
            self.playerFlapAcc = -4.5  # players speed on flapping
            self.playerFlapped = False  # True when player flaps
        else:
            self.pipeVelX = -4         # player velocity, max velocity, downward accleration, accleration on flap
            self.playerVelY    =  -2   # player's velocity along Y, default same as playerFlapped
            self.playerMaxVelY =  10   # max vel along Y, max descend speed
            self.playerMinVelY =  -8   # min vel along Y, max ascend speed
            self.playerAccY    =   1   # players downward accleration
            self.playerFlapAcc =  -9   # players speed on flapping
            self.playerFlapped = False # True when player flaps

    def frame_step(self, input_actions):
        currentScore = self.score
        pygame.event.pump()

        #Added in for difficulty hard, gradually increase the speed
        if self.difficulty == 1:
            self.pipeVelX -= 0.01
            self.playerVelY -= 0.01  # player's velocity along Y, default same as playerFlapped
            if self.playerAccY <= 5:
                self.playerMinVelY -= 0.005  # min vel along Y, max ascend speed
                self.playerAccY += 0.005  # players downward accleration

        reward = 0.1
        terminal = False

        if sum(input_actions) != 1:
            raise ValueError('Multiple input actions!')

        # input_actions[0] == 1: do nothing
        # input_actions[1] == 1: flap the bird
        if input_actions[1] == 1:
            if self.playery > -2 * PLAYER_HEIGHT:
                self.playerVelY = self.playerFlapAcc
                self.playerFlapped = True

        # check for score
        playerMidPos = self.playerx + PLAYER_WIDTH / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + PIPE_WIDTH / 2
            if pipeMidPos <= playerMidPos < pipeMidPos - self.pipeVelX:
                self.score += 1
                reward = 1

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(PLAYER_INDEX_GEN)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False
        self.playery += min(self.playerVelY, BASEY - self.playery - PLAYER_HEIGHT)
        if self.playery < 0:
            self.playery = 0

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        #top variable and second if statement added for different difficulty levels
        top = 1 - self.pipeVelX
        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < top:
            if len(self.upperPipes) <= 2:
                newPipe = getRandomPipe(self.difficulty)
                self.upperPipes.append(newPipe[0])
                self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -PIPE_WIDTH:
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        # check if crash here
        isCrash = checkCrash({'x': self.playerx, 'y': self.playery,
                              'index': self.playerIndex},
                             self.upperPipes, self.lowerPipes)
        if isCrash:
            terminal = True
            self.__init__()
            reward = -1

        if self.score == 20:
            terminal = True
            self.__init__()
            reward = 20

        # draw sprites
        SCREEN.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        SCREEN.blit(IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        # showScore(self.score)
        SCREEN.blit(IMAGES['player'][self.playerIndex],
                    (self.playerx, self.playery))

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)
        return image_data, reward, terminal, currentScore


def getRandomPipe(difficulty):
    """returns a randomly generated pipe"""
    # y of gap between upper and lower pipe
    if difficulty == -1:
        gapYs = [35, 40, 45, 50, 55, 60, 65, 70]
    elif difficulty == 1:
        gapYs = [20, 40, 60, 80, 100, 120, 140, 160]
    else:
        gapYs = [20, 30, 40, 50, 60, 70, 80, 90]

    index = random.randint(0, len(gapYs) - 1)
    gapY = gapYs[index]

    gapY += int(BASEY * 0.2)
    pipeX = SCREENWIDTH + 10

    return [
        {'x': pipeX, 'y': gapY - PIPE_HEIGHT},  # upper pipe
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
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

    # if player crashes into ground
    if player['y'] + player['h'] >= BASEY - 1:
        return True
    else:

        playerRect = pygame.Rect(player['x'], player['y'],
                                 player['w'], player['h'])

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            # upper and lower pipe rects
            uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)
            lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], PIPE_WIDTH, PIPE_HEIGHT)

            # player and upper/lower pipe hitmasks
            pHitMask = HITMASKS['player'][pi]
            uHitmask = HITMASKS['pipe'][0]
            lHitmask = HITMASKS['pipe'][1]

            # if bird collided with upipe or lpipe
            uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
            lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

            if uCollide or lCollide:
                return True

    return False


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
    """Checks if two objects collide and not just their rects"""
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False
