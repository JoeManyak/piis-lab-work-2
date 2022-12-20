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
from pacman import GameState


class ReflexAgent(Agent):
    def getAction(self, gameState: GameState):
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn='betterEvaluationFunction', depth='3'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):

        def minimax(state, depth):
            if (state.isWin() or state.isLose()) or depth == 0:
                return None

            legal_actions = state.getLegalActions(0)
            score = float('-inf')
            action = None

            for current_action in legal_actions:
                next_state = state.generateSuccessor(0, current_action)
                current_score = recursiveMinimax(next_state, depth, 1)
                if current_score > score:
                    score = current_score
                    action = current_action

            return action

        def recursiveMinimax(state, depth, agent):
            if (state.isWin() or state.isLose()) or depth == 0:
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(agent)
            next_agent = (agent + 1) % state.getNumAgents()

            if agent == 0:
                score = float('-inf')
                for cur_action in legal_actions:
                    next_state = state.generateSuccessor(agent, cur_action)
                    score = max(score, recursiveMinimax(next_state, depth, next_agent))
            else:
                score = float('inf')
                for cur_action in legal_actions:
                    next_state = state.generateSuccessor(agent, cur_action)
                    if next_agent == 0:
                        score = min(score, recursiveMinimax(next_state, depth - 1, next_agent))
                    else:
                        score = min(score, recursiveMinimax(next_state, depth, next_agent))

            return score

        return minimax(gameState, self.depth)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        def alphaBeta(state, depth):
            if (state.isWin() or state.isLose()) or depth == 0:
                return None

            alpha = float('-inf')
            beta = float('inf')
            legal_actions = state.getLegalActions(0)

            score = float('-inf')
            action = None
            for cur_action in legal_actions:
                next_state = state.generateSuccessor(0, cur_action)
                cur_score = recursiveAlphaBeta(next_state, depth, 1, alpha, beta)
                if cur_score > score:
                    score = cur_score
                    action = cur_action
                if score > beta:
                    break
                alpha = max(alpha, score)
            return action

        def recursiveAlphaBeta(state, depth, agent, alpha, beta):
            if (state.isWin() or state.isLose()) or depth == 0:
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(agent)
            next_agent = (agent + 1) % state.getNumAgents()

            if agent == 0:
                score = float('-inf')
                for cur_action in legal_actions:
                    next_state = state.generateSuccessor(0, cur_action)
                    score = max(score, recursiveAlphaBeta(next_state, depth, next_agent, alpha, beta))
                    if score > beta:
                        break
                    alpha = max(alpha, score)
            else:
                score = float('inf')
                for cur_action in legal_actions:
                    next_state = state.generateSuccessor(agent, cur_action)
                    if next_agent == 0:
                        score = min(score, recursiveAlphaBeta(next_state, depth - 1, next_agent, alpha, beta))
                    else:
                        score = min(score, recursiveAlphaBeta(next_state, depth, next_agent, alpha, beta))
                    if score < alpha:
                        break
                    beta = min(beta, score)

            return score

        return alphaBeta(gameState, self.depth)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        def expectimax(state, depth):
            if (state.isWin() or state.isLose()) or depth == 0:
                return None

            legal_actions = state.getLegalActions(0)

            action = None
            score = float('-inf')
            for cur_action in legal_actions:
                next_state = state.generateSuccessor(0, cur_action)
                current_score = recursiveExpectimax(next_state, depth, 1)
                if current_score > score:
                    score = current_score
                    action = cur_action
            return action

        def recursiveExpectimax(state, depth, agent):
            if (state.isWin() or state.isLose()) or depth == 0:
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(agent)
            next_agent = (agent + 1) % state.getNumAgents()

            score = 0
            if agent == 0:
                score = float('-inf')
                for current_action in legal_actions:
                    next_state = state.generateSuccessor(agent, current_action)
                    score = max(score, recursiveExpectimax(next_state, depth, next_agent))
            else:
                for current_action in legal_actions:
                    next_state = state.generateSuccessor(agent, current_action)
                    if next_agent == 0:
                        score += recursiveExpectimax(next_state, depth - 1, next_agent)
                    else:
                        score += recursiveExpectimax(next_state, depth, next_agent)
                score /= len(legal_actions)

            return score

        return expectimax(gameState, self.depth)


def betterEvaluationFunction(currentGameState: GameState):
    score = currentGameState.getScore()
    if currentGameState.isLose():
        return -999999 + score
    elif currentGameState.isWin():
        return 999999 + score

    pacman_pos = currentGameState.getPacmanPosition()
    food_left = currentGameState.getFood().asList()
    food_distances = []
    for food in food_left:
        food_distances.append(util.manhattanDistance(pacman_pos, food))
    nearest_food = min(food_distances)

    ghosts = currentGameState.getGhostStates()
    active_ghost_distances = []
    for ghost in ghosts:
        if not ghost.scaredTimer:
            active_ghost_distances.append(util.manhattanDistance(pacman_pos, ghost.getPosition()))
    nearest_ghost = 0
    if len(active_ghost_distances) != 0:
        nearest_ghost = min(active_ghost_distances)

    score_c = 6
    food_left_c = 6
    nearest_food_c = 2
    nearest_ghost_c = 2

    evaluation = score_c * score - food_left_c * len(food_left) - nearest_food_c * nearest_food
    if nearest_ghost != 0:
        evaluation -= nearest_ghost_c * (1. / nearest_ghost)

    return evaluation


better = betterEvaluationFunction
