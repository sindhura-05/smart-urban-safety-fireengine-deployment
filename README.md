# Smart Urban Safety: Fire Engine Deployment using GNNs and Reinforcement Learning

## Overview
This project focuses on optimizing fire engine deployment in urban areas by modeling the city as a graph. Nodes represent locations and edges represent connectivity. The objective is to select critical nodes for deployment to maximize coverage.

## Approach
- Graph Neural Networks (GNNs) learn node embeddings
- Reinforcement Learning (Double Deep Q-Network) selects deployment nodes
- Rewards are based on coverage improvement

## Technologies Used
- Python
- PyTorch
- PyTorch Geometric
- NetworkX

## Files
- graph_utils.py: Graph generation
- gnn_model.py: GNN architecture
- env.py: Reinforcement learning environment
- ddqn_agent.py: DDQN agent
- train.py: Training loop

## How to Run
pip install -r requirements.txt
python train.py

## Results
The agent learns to select nodes that maximize coverage in the graph, demonstrating near-optimal deployment behavior on synthetic urban networks.

## Applications
- Emergency response optimization
- Smart city planning
- Graph-based decision making
