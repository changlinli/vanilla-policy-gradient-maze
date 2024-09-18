#%% 

print("hello!")

# %%

1 + 1

# %%

import torch

# %%

fixed_maze = torch.tensor(
    [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]
)

starting_point = (0, 0)

ending_point = (4, 4)

from enum import Enum

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

# %%

def make_move(agent_state: tuple[int, int], action: Action) -> tuple[int, int]:
    agent_state_x, agent_state_y = agent_state
    match action:
        case Action.UP:
            out_of_bounds = agent_state_y - 1 < 0
            if out_of_bounds:
                return agent_state
            else:
                # Remember to reverse these coordinates because the first index
                # will select the row and the second by column, so if you pass
                # in (x, y) you should index first by y and then x
                wall_in_way = fixed_maze[agent_state_y - 1][agent_state_x] == 1
                if wall_in_way:
                    return agent_state
                else:
                    return agent_state_x, agent_state_y - 1
        case Action.DOWN:
            out_of_bounds = agent_state_y + 1 > 4
            if out_of_bounds:
                return agent_state
            else:
                # Remember to reverse these coordinates because the first index
                # will select the row and the second by column, so if you pass
                # in (x, y) you should index first by y and then x
                wall_in_way = fixed_maze[agent_state_y + 1][agent_state_x] == 1
                if wall_in_way:
                    return agent_state
                else:
                    return agent_state_x, agent_state_y + 1
        case Action.LEFT:
            out_of_bounds = agent_state_x - 1 < 0
            if out_of_bounds:
                return agent_state
            else:
                # Remember to reverse these coordinates because the first index
                # will select the row and the second by column, so if you pass
                # in (x, y) you should index first by y and then x
                wall_in_way = fixed_maze[agent_state_y][agent_state_x - 1] == 1
                if wall_in_way:
                    return agent_state
                else:
                    return agent_state_x - 1, agent_state_y
        case Action.RIGHT:
            out_of_bounds = agent_state_x + 1 > 4
            if out_of_bounds:
                return agent_state
            else:
                # Remember to reverse these coordinates because the first index
                # will select the row and the second by column, so if you pass
                # in (x, y) you should index first by y and then x
                wall_in_way = fixed_maze[agent_state_y][agent_state_x + 1] == 1
                if wall_in_way:
                    return agent_state
                else:
                    return agent_state_x + 1, agent_state_y

print(f"{make_move((0, 1), Action.UP)=}")
print(f"{make_move((0, 1), Action.DOWN)=}")
print(f"{make_move((0, 1), Action.LEFT)=}")
print(f"{make_move((0, 1), Action.RIGHT)=}")
print(f"{make_move((0, 4), Action.RIGHT)=}")

# %%

list(enumerate([1, 2, 3]))

# %%

# Visualize the maze

def visualize(agent_state: tuple[int, int]):
    visualization_string = ""
    for row_index, row in enumerate(fixed_maze):
        current_visualized_row = "|"
        for column_index, column_value in enumerate(row):
            if (column_index, row_index) == agent_state:
                current_visualized_row += " * |"
            else:
                current_visualized_row += f" {column_value} |"
        visualization_string += current_visualized_row + "\n"
    print(visualization_string)

visualize((2, 3))

# %%

# Test that moves work

initial_state = (0, 0)
visualize(initial_state)
new_state_0 = make_move(initial_state, Action.DOWN)
visualize(new_state_0)
new_state_1 = make_move(new_state_0, Action.DOWN)
visualize(new_state_1)
new_state_2 = make_move(new_state_1, Action.RIGHT)
visualize(new_state_2)

# %%

def reward_function(old_state: tuple[int, int], new_state: tuple[int, int]) -> float:
    new_state_x, new_state_y = new_state
    ending_point_x, ending_point_y = ending_point
    distance_from_end = abs(ending_point_x - new_state_x) + abs(ending_point_y - new_state_y)
    if new_state == ending_point:
        return 30.0
    # We ran into a wall and weren't able to move
    elif new_state == old_state:
        return -10
    else:
        return -1.0 - distance_from_end

# %%

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stack = nn.Sequential(
            nn.Linear(in_features=2, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=4),
            nn.Softmax(dim=-1),
        )

    def forward(self, input: Float[Tensor, "state_xy"]) -> Float[Tensor, "action_probs"]:
        return self.stack(input)

starting_network = PolicyNetwork()

# %%

def one_step_of_inference(current_state: tuple[int, int], policy_network: PolicyNetwork) -> tuple[Action, Float[Tensor, ""]]:
    action_probs = policy_network(torch.Tensor(current_state))
    distribution = Categorical(action_probs)
    action_idx = distribution.sample().item()
    return (Action(action_idx), action_probs[action_idx])

one_step_of_inference(starting_point, starting_network)

# %%

from dataclasses import dataclass

@dataclass
class TrajectoryItem:
    state: tuple[int, int]
    action: Action
    action_prob: Float[Tensor, ""]
    reward: float

# %%

MAX_NUM_OF_MOVES = 100

def generate_trajectory(policy_network: PolicyNetwork) -> list[TrajectoryItem]:
    current_position = starting_point
    number_of_moves = 0
    current_trajectory = []
    while number_of_moves < MAX_NUM_OF_MOVES:
        new_action, new_action_prob = one_step_of_inference(current_position, policy_network)
        new_position = make_move(current_position, new_action)
        reward = reward_function(old_state=current_position, new_state=new_position)
        current_trajectory.append(TrajectoryItem(state=current_position, action=new_action, action_prob=new_action_prob, reward=reward))
        if new_position == ending_point:
            break
        current_position = new_position
        number_of_moves += 1
    return current_trajectory

example_trajectory = generate_trajectory(starting_network)
example_trajectory

# %%

def calculate_total_rewards(trajectory: list[TrajectoryItem]) -> list[tuple[tuple[int, int], float, Float[Tensor, ""]]]:
    total_rewards = []
    for trajectory_idx, trajectory_item in enumerate(trajectory):
        state = trajectory_item.state
        action_prob = trajectory_item.action_prob
        future_rewards = [ trajectory_item.reward for trajectory_item in trajectory[trajectory_idx:] ]
        total_rewards.append((state, action_prob, sum(future_rewards) / 100))
    return total_rewards

example_total_rewards = calculate_total_rewards(example_trajectory)
example_total_rewards

# %%

def calculate_loss_from_total_rewards(total_rewards: list[tuple[tuple[int, int], float, Float[Tensor, ""]]]) -> Float[Tensor, ""]:
    current_loss = Tensor([0.0])
    for state, action_prob, total_reward in total_rewards:
        current_loss += action_prob * total_reward
    return current_loss

calculate_loss_from_total_rewards(example_total_rewards)

# %%

from torch.optim import SGD

MAX_TRAINING_ITERATIONS = 5000

new_network = PolicyNetwork()

# %%

def train(policy_network: PolicyNetwork):
    training_iteration = 0
    optimizer = SGD(policy_network.parameters(), lr=0.01, momentum=0.9)
    while training_iteration < MAX_TRAINING_ITERATIONS:
        print(f"{training_iteration=}")
        optimizer.zero_grad()
        trajectory = generate_trajectory(policy_network)
        total_rewards = calculate_total_rewards(trajectory)
        loss = calculate_loss_from_total_rewards(total_rewards) * -1
        loss.backward()
        optimizer.step()
        training_iteration += 1

train(new_network)

# %%

generate_trajectory(new_network)