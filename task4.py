import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

# CODTECH Task 4: Reinforcement Learning for Tic-Tac-Toe
# Agent learns by playing against itself

class TicTacToe:
    def __init__(self):
        self.board = [' ']*9
        self.winner = None

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, position, player):
        if self.board[position] == ' ':
            self.board[position] = player
            return True
        return False

    def check_winner(self):
        win_combos = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        for a,b,c in win_combos:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                self.winner = self.board[a]
                return self.winner
        if ' ' not in self.board:
            self.winner = 'Draw'
            return 'Draw'
        return None

    def get_state(self):
        return ''.join(self.board)

# Q-Learning Agent
class QAgent:
    def __init__(self, epsilon=0.2, alpha=0.3, gamma=0.9):
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            q_values = [self.get_q(state, a) for a in available_actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_actions):
        best_next_q = max([self.get_q(next_state, a) for a in next_actions], default=0)
        current_q = self.get_q(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)
        self.q_table[(state, action)] = new_q

# Training
def train(episodes=20000):
    agent = QAgent()
    rewards_history = []
    
    for ep in range(episodes):
        game = TicTacToe()
        state_history = []
        player = 'X'
        
        while True:
            state = game.get_state()
            actions = game.available_moves()
            if not actions:
                break
            
            action = agent.choose_action(state, actions)
            state_history.append((state, action, player))
            
            game.make_move(action, player)
            winner = game.check_winner()
            
            if winner:
                # Reward system
                for s, a, p in state_history:
                    if winner == 'Draw':
                        reward = 0.5
                    elif p == winner:
                        reward = 1 if p == 'X' else -1
                    else:
                        reward = -1 if p == 'X' else 1
                    
                    next_state = game.get_state()
                    next_actions = game.available_moves()
                    agent.update(s, a, reward, next_state, next_actions)
                break
            
            player = 'O' if player == 'X' else 'X'
        
        if ep % 1000 == 0:
            rewards_history.append(len(agent.q_table))
            print(f"Episode {ep}/{episodes} - Q-Table size: {len(agent.q_table)}")
    
    return agent, rewards_history

# Run Training
print("Training RL Agent for Tic-Tac-Toe...")
trained_agent, history = train(episodes=10000)

# Save model
with open('rl_agent.pkl', 'wb') as f:
    pickle.dump(trained_agent.q_table, f)

print(f"\nTraining Complete! Q-Table learned {len(trained_agent.q_table)} state-action values")
print("Model saved as rl_agent.pkl")

# Plot learning progress
plt.plot(history)
plt.xlabel('Training Progress (x1000 episodes)')
plt.ylabel('Q-Table Size (Knowledge)')
plt.title('Reinforcement Learning Progress')
plt.savefig('rl_learning_graph.png')
plt.show()

# Demo: AI vs AI
print("\nDemo Game: Trained Agent X vs O")
game = TicTacToe()
player = 'X'
while not game.check_winner():
    state = game.get_state()
    actions = game.available_moves()
    if not actions:
        break
    if player == 'X':
        action = trained_agent.choose_action(state, actions)
    else:
        action = random.choice(actions)
    game.make_move(action, player)
    print(f"Player {player} moved to {action}")
    print(game.board[0:3])
    print(game.board[3:6])
    print(game.board[6:9])
    print("-----")
    player = 'O' if player == 'X' else 'X'

print(f"Result: {game.winner}")
