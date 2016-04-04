# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
        #A capable reflex agent will have to consider both food locations and ghost locations to perform well.
        #Your agent should easily and reliably clear the testClassic layout

    def closestDotDistance(self, pos1, foodlist):
        min = 0
        for food in foodlist:
            distance = manhattanDistance(pos1, food)
            if distance > min:
                min = distance
        return min

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #chosenIndex = bestIndices[0]

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        currentFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        walls = successorGameState.data.layout.walls.data
        "*** YOUR CODE HERE ***"

        score = 0
        x,y = newPos

        r = 2
        w = newFood.width
        h = newFood.height

        if action == 'Stop':
            score -= 50
        # elif action == 'North':
        #     for n in range(y, h):
        #         if walls[x][n]:
        #             break
        #         if newFood[x][n]:
        #             score += 1
        # elif action == 'East':
        #     for e in range(x, w):
        #         if walls[e][y]:
        #             break
        #         if newFood[e][y]:
        #             score += 1
        # elif action == 'West':
        #     for w in range(x, 0, -1):
        #         if walls[w][y]:
        #             break
        #         if newFood[w][y]:
        #             score += 1
        # elif action == 'South':
        #     for s in range(y, 0, -1):
        #         if walls[x][s]:
        #             break
        #         if newFood[x][s]:
        #             score += 1


        if currentFood[x][y]:
            score += 9999
        else:
            score -= closestDot(newPos, currentFood)

        # capsules = currentGameState.data.capsules
        # for capsule in capsules:
        #     cx, cy = capsule
        #     score += manhattanDistance((cx, cy), newPos)

        for ghost in newGhostStates:
            gx, gy = ghost.configuration.pos
            if gx == x and gy == y:
                score -= 99999999999999
            ghostPenalty = manhattanDistance(newPos, ghost.configuration.pos)
            #score += ghostPenalty
            direction = ghost.configuration.direction

        print 'Action: ', action
        print 'Score: ', score
        print'----------------'
        return score



def closestDot(pos1, foodlist):
    min = 99999
    rx = 0
    ry = 0

    for x in range(0, foodlist.width):
        for y in range(0, foodlist.height):
            if foodlist[x][y]:
                distance = euclideanDistance(pos1, (x, y))
                # print distance, min
                if distance < min:
                    min = distance
                    rx = x
                    ry = y

    print "Furthest: ",rx ,ry
    print "Distance: ", min
    return min

def euclideanDistance(position1, position2):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position1
    xy2 = position2
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def value(self, state):
        print 'test'


    def minvalue(self, state):
        v = -9999999999
        for successor in state.getLegalActions(1):
            v = min(v, self.evaluationFunction(successor))
        return v

    def maxvalue(self, state):
        v = 9999999999
        for successor in state.getLegalActions(0):
            v = max(v, self.evaluationFunction(successor))
        return v


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        depth = self.depth
        index = self.index

        #For all possible actions, get minimax









        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

