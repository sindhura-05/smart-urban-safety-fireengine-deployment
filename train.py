from graph_utils import generate_graph
from env import FireEnv
from ddqn_agent import Agent
from gnn_model import GNN
import torch

G, x, edge_index = generate_graph(10)

env = FireEnv(G)
agent = Agent(G.number_of_nodes())

gnn = GNN(x.shape[1], 16)
optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)

for episode in range(50):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        embeddings = gnn(x, edge_index)
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.train(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    print(f"Episode {episode}, Reward: {total_reward}")
