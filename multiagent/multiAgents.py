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

        # print 'Action: ', action
        # print 'Score: ', score
        # print'----------------'
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

    #print "Furthest: ",rx ,ry
    #print "Distance: ", min
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

    def minvalue(self, state, agent, depth):
        v = 99999999999
        for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, action)
            v = min(v, self.minimax(successor, agent+1, depth+1))
        return v

    def maxvalue(self, state, agent, depth):
        v = -999999999
        for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, action)
            v = max(v, self.minimax(successor, agent+1, depth+1))
        return v


    def minimax(self, state, currentAgent, depth):

        if currentAgent == self.ghosts + 1:
            currentAgent = 0

        if depth == (self.depth * (self.ghosts + 1)) or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if currentAgent == 0:
            return self.maxvalue(state, currentAgent, depth)
        else:
            return self.minvalue(state, currentAgent, depth)



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
        actions = {}
        depth = 0
        agent = self.index
        self.ghosts = gameState.getNumAgents() - 1

        # Separate MAX layer to pull out actions.
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            actions[self.minimax(successor, agent + 1, depth + 1)] = action
        return actions[max(actions)]

        #util.raiseNotDefined()

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

    def expvalue(self, state, agent, depth):
        v = 0
        for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, action)
            v += self.expectimax(successor, agent + 1, depth + 1)
        return v / len(state.getLegalActions(agent))

    def maxvalue(self, state, agent, depth):
        v = -999999999
        for action in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, action)
            v = max(v, self.expectimax(successor, agent + 1, depth + 1))
        return v

    def expectimax(self, state, currentAgent, depth):

        if currentAgent == self.ghosts + 1:
            currentAgent = 0

        if depth == (self.depth * (self.ghosts + 1)) or state.isWin() or state.isLose():
            return self.evaluationFunction(state)

        if currentAgent == 0:
            return self.maxvalue(state, currentAgent, depth)
        else:
            return self.expvalue(state, currentAgent, depth)


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = {}
        depth = 0
        agent = self.index
        self.ghosts = gameState.getNumAgents() - 1

        # Separate MAX layer to pull out actions.
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            actions[self.expectimax(successor, agent + 1, depth + 1)] = action
        return actions[max(actions)]
        #util.raiseNotDefined()

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

