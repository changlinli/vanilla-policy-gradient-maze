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
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
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
            out_of_bounds = agent_state_y + 1 < 0
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
            out_of_bounds = agent_state_x + 1 < 0
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