import random
from typing import Tuple, List, Optional
import heapq
class World:

  def __init__(self, size, grid=None):
    # Using a stack structure within a 2D array.
    # Initialize an array of 10 empty arrays.
    self.size = size
    self.grid = grid
    # If no grid is provided, grid=None, so generate a random grid.
    if not self.grid:
      self.gen_random_grid()
    else:
      # Otherwise, check if the custom grid is valid.
      self.handle_custom_grid()

  # Generate a random grid.
  def gen_random_grid(self):
    # Start with 10 empty lists.
    self.grid = [[] for i in range(self.size)]
    # Shuffle the range from 0-9 before randomly adding them to the grid.
    numbers = [i for i in range(self.size)]
    random.shuffle(numbers)
    for num in numbers:
      self.grid[random.randint(0, self.size - 1)].append(num)

  # Error handling for custom grids.
  # Checks if the grid has too many elements, too few elements, if there are duplicates,
  # and if the values are between 0-9. If any of those are False, generate a random grid.
  def handle_custom_grid(self):
      valid_set = set((range(self.size)))
      custom_set = set()
      grid_items = 0
      for col in self.grid:
        grid_items += len(col)
        if grid_items > self.size:
          print(grid_items)
          print(self.size)
          print("Custom list has too many elements. Generating random grid.")
          self.gen_random_grid()
          return
        for value in col:
          if value not in valid_set:
            print("Values must be 0-9. Generating random grid.")
            self.gen_random_grid()
            return
          else:
            if value not in custom_set:
              custom_set.add(value)
            else:
              print("Cannot have duplicate block values. Generating random grid.")
              self.gen_random_grid()
              return

      # If the sets are different, then the custom grid is invalid.
      # Could also check using the grid_items variable.
      if len(valid_set.difference(custom_set)) != 0:
        print("Grids have different number of elements. Generating random grid.")
        self.gen_random_grid()
        return

  # Print out the current state of the grid.
  # Kind of like a 90 degree counter-clockwise rotation on the original list of lists.
  def show(self):
    ret = []
    for row in range(self.size):
      curr = ["|"]
      for col in range(self.size):
        try:
          curr.append(str(self.grid[col][row]) + "|")
        except:
          curr.append(" |")
      ret.append("".join(curr))

    for i in range(len(ret) - 1, -1, -1):
      print(ret[i])
    print("_____________________")

  # Make a random move of one block.
  # Only the blocks on the top of a stack may be moved to any other stack.
  def move_random_block(self):
    # Get all indices of non-empty columns
    non_empty = [i for i, row in enumerate(self.grid) if row]
    # Pop one block from a non-empty column and move it to anywhere except its current position
    col_to_pop = random.choice(non_empty)
    pop_block = self.grid[col_to_pop].pop()
    non_empty.remove(col_to_pop)
    # Append pop_block to a random column
    col_to_add = random.choice([i for i in range(self.size) if i != col_to_pop])
    self.grid[col_to_add].append(pop_block)

  def verify(self, goal_state: Tuple[Tuple[int, ...], ...]) -> bool:
    return self.grid_as_tuple() == goal_state

  def grid_as_tuple(self) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(col) for col in self.grid)


def get_successors(state: Tuple[Tuple[int, ...]]) -> List[Tuple[Tuple[int, ...], Tuple[int, int]]]:
    """
    Generates all possible next states by moving the top block from one stack to another.
    
    Parameters:
    - state: Current state of the world grid.

    Returns:
    - A list of tuples where each tuple contains:
        1. The new state of the grid
        2. A tuple representing the move as (from_column, to_column)
    """
    # Assisted by ChatGPT
    successors = []
    num_columns = len(state)
    for from_col, stack in enumerate(state):
        # If the stack is empty, skip it
        if not stack:
            continue

        stack_top = stack[-1]

        for to_col in range(num_columns):
            if from_col == to_col:
                continue
            # Create a new state by moving the block
            new_state = list(list(col) for col in state)  # Deep copy of the state
            new_state[from_col].pop()  # Remove the block from the current stack
            new_state[to_col].append(stack_top)  # Add the block to the target stack
            new_state_tuple = tuple(tuple(col) for col in new_state)  # Convert back to tuple of tuples

            move = (from_col, to_col)
            successors.append((new_state_tuple, move))
    
    return successors
            
          
# TODO: Implement Node class
# TODO: Implement A* algorithm
# TODO: Implement helper functions for the heuristic
# TODO: sometimes the algorithm will get stuck in a loop, so once A star is implemented, we can add a check to see if the same state has been visited before
# This should get rid of the issue I think.

### REVISIT
def get_support(state: Tuple[Tuple[int, ...], ...], block: int) -> str:
    """
    Returns the block that is directly below the given block in the state.
    If the block is on the table, returns 'Table'.
    """
    for stack in state:
        for i, b in enumerate(stack):
            if b == block:
                if i == 0:
                    return 'Table'
                else:
                    return str(stack[i - 1])
    return 'Table'  # If block not found, assume it's on the table

def heuristic_h1_enhanced(current_state: Tuple[Tuple[int, ...], ...],
                          goal_state: Tuple[Tuple[int, ...], ...],
                          blocks: Tuple[int, ...]) -> int:
    """
    Enhanced heuristic that considers misplaced blocks and dependencies.
    """
    misplaced = 0
    for block in blocks:
        current_support = get_support(current_state, block)
        goal_support = get_support(goal_state, block)
        if current_support != goal_support:
            misplaced += 1
            # Check if any block is on top of this block
            for stack in current_state:
                if block in stack:
                    index = stack.index(block)
                    if index < len(stack) - 1:
                        # Blocks above this block are blocking it
                        misplaced += len(stack) - index - 1
                    break
    return misplaced


def define_goal(world: Tuple[Tuple[int,...]], world_size: int) -> Tuple[Tuple[int, ...], ...]:
    # If the 0 is already at the bottom of a stack, then the goal state is that stack.
    # Otherwise, the goal state is any column that is not currently occupied.
    goal = []
    last_free = None
    found_col = False
    # Sorted order of the blocks
    goal_state = tuple(range(world_size))
    for i, col in enumerate(world.grid):
        # If the column is empty, then it is a free column
        if not col:
            # TODO: Only really need the first free column
            last_free = i
            goal.append(tuple())
            continue
        if col[0] == 0:
            goal.append(goal_state)
            found_col = True
        else:
            goal.append(tuple())
    
    if not found_col:
        goal[last_free] = goal_state    
    return tuple(goal)

def main():
    blocks = tuple(range(6))  # Blocks 0 through 5

    # Initialize the World with fewer columns
    world = World(size=6)  # Reduced from 10 to 5 for testing purposes
    print("Initial World State:")
    world.show()

    # Define the Goal State
    goal_state = define_goal(world, world.size)
    print("Goal State:")
    goal_world = World(grid=[list(col) for col in goal_state], size=6)
    goal_world.show()
    print(goal_world.verify(goal_state))

    start_state = world
    state = start_state
    # while True:
    for i in range(10):
        successors = get_successors(state.grid_as_tuple())
        heuristic_values = []
        for successor in successors:
            heuristic_values.append(heuristic_h1_enhanced(successor[0], goal_state, blocks))
        # print(min(heuristic_values))
        new_state_tuple = successors[heuristic_values.index(min(heuristic_values))][0]
        state = World(grid=[list(col) for col in new_state_tuple], size=6)
        state.show()
        if state.verify(goal_state):
            break
    # print(get_successors(start_state))

    # # Convert World grid to tuple of tuples
    # start_state = world.grid_as_tuple()

    # # Run A* algorithm
    # solution = a_star(start_state, goal_state, blocks)

    # if solution:
    #     print("\nSolution found with {} moves:".format(len(solution)-1))
    #     for step, state in enumerate(solution):
    #         print(f"Step {step}:")
    #         temp_world = World(grid=[list(col) for col in state], num_columns=10)
    #         temp_world.show()
    # else:
    #     print("No solution found.")

if __name__ == "__main__":
    main()