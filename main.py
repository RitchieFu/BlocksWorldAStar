import cProfile
import heapq
import random
import time
from functools import lru_cache
from typing import Tuple, List, Optional

class World:

  def __init__(self, size: int = None, grid: List[List[int]] = None):
    """ 
    The grid is implemented as a list of lists, where each list is a stack/column.
    The size is flexible, although the default value is 10.
    If not grid is provided, generate a random grid. Otherwise, check if the custom grid is valid.
    """
    self.size = size
    if not self.size:
       self.size = 10
    
    self.grid = grid
    if not self.grid:
      self.gen_random_grid()
    else:
      self.handle_custom_grid()


  def gen_random_grid(self):
    """
    Generate a random grid of size self.size.
    This function may be used after a solution has been found to generate a new random grid.
    """
    # Start with 10 empty lists.
    self.grid = [[] for _ in range(self.size)]
    # Shuffle the range from 0-9 before randomly adding them to the grid.
    numbers = [i for i in range(self.size)]
    random.shuffle(numbers)
    for num in numbers:
      self.grid[random.randint(0, self.size - 1)].append(num)

  
  def handle_custom_grid(self):
    """
    Error handling for custom grids.
    Checks if the grid has too many elements, too few elements, if there are duplicates,
    or if the values are between 0-9. If any of those are true, generate a random grid.
    """
    valid_set = set((range(self.size)))
    custom_set = set()
    grid_items = 0
    for col in self.grid:
      grid_items += len(col)
      # Too many elements in the custom grid
      if grid_items > self.size:
        print(grid_items)
        print(self.size)
        print("Custom list has too many elements. Generating random grid.")
        self.gen_random_grid()
        return
      for value in col:
        # Value not in the range of 0 to self.size
        if value not in valid_set:
          print("Values must be 0-9. Generating random grid.")
          self.gen_random_grid()
          return
        else:
          if value not in custom_set:
            custom_set.add(value)
          else:
            # Duplicate values in the custom grid
            print("Cannot have duplicate block values. Generating random grid.")
            self.gen_random_grid()
            return

    # If the code reaches this point, the values are valid but the sets may have different lengths.
    if len(valid_set.difference(custom_set)) != 0:
      print("Grids have different number of elements. Generating random grid.")
      self.gen_random_grid()
      return


  def show(self):
    """
    Print out the current state of the grid.
    Kind of like a 90 degree counter-clockwise rotation of the grid.
    """
    ret = []
    for row in range(self.size):
      curr = ["|"]
      for col in range(self.size):
        try:
          curr.append(str(self.grid[col][row]) + "|")
        except: # If the column is empty, print a space.
          curr.append(" |")
      ret.append("".join(curr))

    # Print the grid in reverse order so that the 1st row is at the bottom.
    for i in range(len(ret) - 1, -1, -1):
      print(ret[i])
    print("_____________________")

  
  def move_random_block(self):
    """
    Make a random move of one block.
    Only the blocks on the top of a stack may be moved to any other stack.
    """
    # Get all indices of non-empty columns
    non_empty = [i for i, row in enumerate(self.grid) if row]
    # Pop one block from a non-empty column and move it to anywhere except its current position
    col_to_pop = random.choice(non_empty)
    pop_block = self.grid[col_to_pop].pop()
    # Line below does nothing. Considering removing it.
    # non_empty.remove(col_to_pop)
    # Append pop_block to a random column
    col_to_add = random.choice([i for i in range(self.size) if i != col_to_pop])
    self.grid[col_to_add].append(pop_block)


  def verify(self, goal_state: Tuple[Tuple[int, ...], ...]) -> bool:
    """Check if the current state of the grid is the same as the goal state."""
    return self.grid_as_tuple() == goal_state


  def grid_as_tuple(self) -> Tuple[Tuple[int, ...], ...]:
    """
    Convert the grid to a tuple so that we can add it to the closed set.
    The closed set is used to keep track of visited states.
    """
    return tuple(tuple(col) for col in self.grid)

  
class Node:
  def __init__(self, state: Tuple[Tuple[int, ...], ...], parent: Optional['Node'],
                cost: int, heuristic: int, last_move: Optional[Tuple[int, int]] = None):
      
    self.state = state # This is the state of the Node as a tuple of tuple of integers.
    self.parent = parent # Keep track of the parent node so we can reconstruct the path later.
    self.cost = cost  # g(n): Cost from start to get to the current node
    self.heuristic = heuristic  # h(n): Estimated cost to goal
    self.last_move = last_move  # (from_col, to_col)

  def __lt__(self, other: 'Node'):
    # Need this operator for the priority queue/min heap so that we can explore the nodes with the lowest cost first.
    # Nodes are compared based on the total cost of f(n) = g(n) + h(n)
    return (self.cost + self.heuristic) < (other.cost + other.heuristic)
      
      
# Helper Functions
def get_successors(state: Tuple[Tuple[int, ...]]) -> List[Tuple[Tuple[int, ...], Tuple[int, int]]]:
    """
    Generates all possible next states by moving the top block from one stack to another.
    The function takes in a grid state and returns a list of possible successor states.
    In the return list, each tuple contains:
        1. The new state of the grid (tuple of tuples of integers)
        2. A tuple representing the move as (from_column, to_column)
    """
    successors = []
    num_columns = len(state)
    for from_col, stack in enumerate(state):
        # If the stack is empty, skip it
        if not stack:
            continue
        
        # Get the top block of the stack
        stack_top = stack[-1]

        # Go through all the other columns except the current one
        for to_col in range(num_columns):
            if from_col == to_col:
                continue
            # Create a new state by moving the block
            # Copy all of the elements in state and convert it to a list of lists
            # Must do this because tuples are immutable
            new_state = list(list(col) for col in state)  
            new_state[from_col].pop()  # Remove the block from the current stack
            new_state[to_col].append(stack_top)  # Add the block to the target stack
            new_state_tuple = tuple(tuple(col) for col in new_state)  # Convert back to tuple of tuples

            move = (from_col, to_col)
            successors.append((new_state_tuple, move))
    
    return successors
   
            
def a_star(start_state: Tuple[Tuple[int, ...], ...], 
           goal_state: Tuple[Tuple[int, ...], ...],
           blocks_in_grid: Tuple[int, ...]) -> Optional[List[Tuple[Tuple[int, ...], ...]]]:
   
    # The open set is a priority queue of nodes to explore, with lowest cost nodes explored first
    open_set = []
    # Calculate the heuristic for the initial state
    start_h = heuristic(start_state, blocks_in_grid)
    heapq.heappush(open_set, Node(start_state, None, 0, start_h))
    # These are the states that have already been visited. Since states are tuples, they can be hashed and added to a set.
    # We do not want to revisit states as sometimes this can cause infinite loops or extra steps.
    closed_set = set()

    while open_set:
        # Get the node with the lowest cost
        current_node = heapq.heappop(open_set)

        if current_node.state == goal_state:
            return reconstruct_path(current_node)

        # If this state has already been visited, move on to the next one
        if current_node.state in closed_set:
            continue
        
        # Add the current state to the closed set
        closed_set.add(current_node.state)

        # Generate all possible next moves from the current state
        for successor, move in get_successors(current_node.state):

            # If a next state has already been visited, skip it
            if successor in closed_set:
              continue

            # If a node has a last move and the current move is just the last move reversed, skip it
            if current_node.last_move and move == (current_node.last_move[1], current_node.last_move[0]):
                continue

            h = heuristic(successor, blocks_in_grid)
            successor_node = Node(successor, current_node, current_node.cost + 1, h, move)
            heapq.heappush(open_set, successor_node)

    # No solution found
    return None  


def get_block_below(state: Tuple[Tuple[int, ...], ...], block: int) -> int:
    """
    Returns the block that is directly below the given block in the state.
    If the block is on the table, return -1.
    """
    for stack in state:
        for i, b in enumerate(stack):
            if b == block:
                if i == 0:
                    return -1
                else:
                    return stack[i - 1]
    return -1  # If block not found, assume it's on the table




def define_goal(world: Tuple[Tuple[int,...]], world_size: int) -> Tuple[Tuple[int, ...], ...]:
  """
  Find the first open column for the goal state.
  If the 0 is already at the bottom of a stack, then the goal state starts at that stack.
  
  I was co
  nsidering optimizing where the goal state would start by looking at the number of blocks on top of 0.
  If there are blocks on top of 0, then the goal state would be the nth free column after the first free column.
  I considered this because I noticed that for some cases, trying to get 0 in the first free column
  slowed down the search or added extra moves.
  """
  # Set up a blank goal state, eventually will populate with the blocks in order
  goal = [tuple() for _ in range(world_size)]
  first_free_col = -1
  found_zero_bottom = False
  # Sorted order of the blocks
  goal_state = tuple(range(world_size))
  for i, col in enumerate(world.grid):
    # If the column is empty, then it is a free column
    if not col: 
      if first_free_col == -1:
        first_free_col = i
    else:
      # If the bottom of the stack is a 0, then you have found the goal state
      # Then, you can break and return the goal state.
      if col[0] == 0:
        goal[i] = goal_state
        found_zero_bottom = True
        break
  
  # If none of the rows had 0 on the bottom, then the goal state starts at the first free column
  if not found_zero_bottom:
    goal[first_free_col] = goal_state    
  
  return tuple(goal)


def reconstruct_path(node: Node) -> List[Node]:
  """
  Takes a node and goes up the parent tree to reconstruct the path.
  """
  path = []
  while node:
    path.append(node)
    node = node.parent
  path.reverse()
  return path
  
def heuristic(current_state: Tuple[Tuple[int, ...], ...], all_blocks: Tuple[int, ...]) -> int:
  """
  This heuristic looks at each block.
  If the block below a block is not correct, then we know that block is misplaced.
  However, it is not enough to just count the number of misplaced blocks.
  Some blocks are restricted by other blocks, so it takes more moves to get them to the correct position.
  I calculate that by counting the number of blocks above the misplaced block.
  
  """
  misplaced = 0
  
  for block in all_blocks:
      current_support = get_block_below(current_state, block)
      goal_support = block - 1
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


def main():
    grid_size = 10
    blocks = tuple(range(grid_size))  # Blocks 0 through 5
    flag = False
    world = World(size=grid_size, grid=[[1,0,2,3,4,5,6,7,8,9], [], [], [], [], [], [], [], [], []])
    # world.show()
    # world = World(size=grid_size)
    goal_state = None
    start_state = None
    solution = None
    for i in range(1):
      print(i)
      print("Initial World State:")
      print(world.grid)
      # world.show()

      # Define the Goal State
      goal_state = define_goal(world, world.size)
      print(goal_state)
      start_state = world.grid_as_tuple()
      start = time.time()
      solution = a_star(start_state, goal_state, blocks)
      end = time.time()
      if solution:
          print("Solution found in {:.2f} seconds with {} moves:".format(end - start, len(solution)-1))
          print("\nSolution found with {} moves:".format(len(solution)-1))
          for step, state in enumerate(solution):
            print(f"Step {step}:")
            print(f"Heuristic estimate: {state.heuristic}")
            temp_world = World(size=grid_size, grid=[list(col) for col in state.state])
            temp_world.show()
          world.gen_random_grid()
          continue
      else:
          print(start_state)
          print("No solution found.")
          flag = True
          break
    
    if not flag:
       print("A* worked for all 100 cases")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.print_stats(sort='time')
