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

def make_move(agent_state: tuple[int, int], action: Action) -> tuple[tuple[int, int], float]:
    agent_state_x, agent_state_y = agent_state
    # small negative reward for each timestep, can end game at e.g. -100
    # large positive reward for reaching end of game
    match action:
        case Action.UP:
            out_of_bounds = agent_state_y - 1 < 0
            if out_of_bounds:
                new_state = agent_state
            else:
                # Remember to reverse these coordinates because the first index
                # will select the row and the second by column, so if you pass
                # in (x, y) you should index first by y and then x
                wall_in_way = fixed_maze[agent_state_y - 1][agent_state_x] == 1
                if wall_in_way:
                    new_state = agent_state
                new_state = agent_state_x, agent_state_y - 1
        case Action.DOWN:
            out_of_bounds = agent_state_y + 1 > 4
            if out_of_bounds:
                new_state = agent_state
            else:
                # Remember to reverse these coordinates because the first index
                # will select the row and the second by column, so if you pass
                # in (x, y) you should index first by y and then x
                wall_in_way = fixed_maze[agent_state_y + 1][agent_state_x] == 1
                if wall_in_way:
                    new_state = agent_state
                new_state = agent_state_x, agent_state_y + 1
        case Action.LEFT:
            out_of_bounds = agent_state_x - 1 < 0
            if out_of_bounds:
                new_state = agent_state
            else:
                # Remember to reverse these coordinates because the first index
                # will select the row and the second by column, so if you pass
                # in (x, y) you should index first by y and then x
                wall_in_way = fixed_maze[agent_state_y][agent_state_x - 1] == 1
                if wall_in_way:
                    new_state = agent_state
                new_state = agent_state_x - 1, agent_state_y
        case Action.RIGHT:
            out_of_bounds = agent_state_x + 1 > 4
            if out_of_bounds:
                new_state = agent_state
            else:
                # Remember to reverse these coordinates because the first index
                # will select the row and the second by column, so if you pass
                # in (x, y) you should index first by y and then x
                wall_in_way = fixed_maze[agent_state_y][agent_state_x + 1] == 1
                if wall_in_way:
                    new_state = agent_state
                else:
                    new_state = agent_state_x + 1, agent_state_y

    if new_state == (4,4):
        return (new_state, 100)
    else:
        return (new_state, -1)

print(f"{make_move((0, 1), Action.UP)=}")
print(f"{make_move((0, 1), Action.DOWN)=}")
print(f"{make_move((0, 1), Action.LEFT)=}")
print(f"{make_move((0, 1), Action.RIGHT)=}")
print(f"{make_move((0, 4), Action.RIGHT)=}")
print(f"{make_move((4, 3), Action.DOWN)=}")