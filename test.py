import random
from typing import Tuple, List, Optional
import heapq
from functools import lru_cache

class World:
    def __init__(self, grid: Optional[List[List[int]]] = None, num_columns: int = 5):
        self.num_columns = num_columns
        self.grid = grid
        if not self.grid:
            self.gen_random_start()
        else:
            self.handle_custom_grid()

    def heuristic(self, goal_state: Tuple[Tuple[int, ...], ...]) -> int:
        # Placeholder, not used in this implementation
        pass

    def gen_random_start(self):
        self.grid = [[] for _ in range(self.num_columns)]
        numbers = list(range(10))
        random.shuffle(numbers)
        for num in numbers:
            self.grid[random.randint(0, self.num_columns - 1)].append(num)

    def handle_custom_grid(self):
        valid_set = set(range(10))
        custom_set = set()
        grid_items = 0
        for col in self.grid:
            grid_items += len(col)
            if grid_items > 10:
                print("Custom list has too many elements. Generating random grid.")
                self.gen_random_start()
                return
            for value in col:
                if value not in valid_set:
                    print("Values must be 0-9. Generating random grid.")
                    self.gen_random_start()
                    return
                if value in custom_set:
                    print("Cannot have duplicate block values. Generating random grid.")
                    self.gen_random_start()
                    return
                custom_set.add(value)

        if len(valid_set.difference(custom_set)) != 0:
            print("Grids have different number of elements. Generating random grid.")
            self.gen_random_start()
            return

    def show(self):
        ret = []
        max_height = max(len(col) for col in self.grid) if self.grid else 0
        for row in range(max_height):
            curr = ["|"]
            for col in range(self.num_columns):
                try:
                    curr.append(str(self.grid[col][row]) + "|")
                except IndexError:
                    curr.append(" |")
            ret.append("".join(curr))

        for i in range(len(ret) - 1, -1, -1):
            print(ret[i])
        print("_____________________")

    def move_random_block(self):
        non_empty = [i for i, row in enumerate(self.grid) if row]
        if not non_empty:
            return  # No move possible
        col_to_pop = random.choice(non_empty)
        pop_block = self.grid[col_to_pop].pop()
        non_empty.remove(col_to_pop)
        col_to_add = random.choice([i for i in range(self.num_columns) if i != col_to_pop])
        self.grid[col_to_add].append(pop_block)

    def verify(self, goal_state: Tuple[Tuple[int, ...], ...]) -> bool:
        return self.grid_as_tuple() == goal_state

    def grid_as_tuple(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(col) for col in self.grid)

class Node:
    def __init__(self, state: Tuple[Tuple[int, ...], ...], parent: Optional['Node'],
                 cost: int, heuristic: int, last_move: Optional[Tuple[int, int]] = None):
        self.state = state
        self.parent = parent
        self.cost = cost  # g(n): Cost from start to current node
        self.heuristic = heuristic  # h(n): Estimated cost to goal
        self.last_move = last_move  # (from_col, to_col)

    def __lt__(self, other: 'Node'):
        # Nodes are compared based on f(n) = g(n) + h(n)
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


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

@lru_cache(maxsize=None)
def get_support_cached(state: Tuple[Tuple[int, ...], ...], block: int) -> str:
    return get_support(state, block)

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

from typing import List, Tuple

def get_successors(state: Tuple[Tuple[int, ...], ...],
                  blocks: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], Tuple[int, int]]]:
    """
    Generates all possible successor states by moving the top block from one stack to another.
    
    Parameters:
    - state: Current state represented as a tuple of tuples.
    - blocks: Tuple containing all block identifiers.
    
    Returns:
    - List of tuples where each tuple contains:
        1. The new state after the move.
        2. A tuple representing the move as (from_column, to_column).
    """
    successors = []
    num_columns = len(state)

    for from_col, stack in enumerate(state):
        if not stack:
            continue  # Skip empty stacks

        block_to_move = stack[-1]  # Only the top block can be moved

        for to_col in range(num_columns):
            if from_col == to_col:
                continue  # Cannot move to the same column

            # Create a new state by moving the block
            new_state = list(list(col) for col in state)  # Deep copy of the state
            new_state[from_col].pop()  # Remove the block from the current stack
            new_state[to_col].append(block_to_move)  # Add the block to the target stack
            new_state_tuple = tuple(tuple(col) for col in new_state)  # Convert back to tuple of tuples

            move = (from_col, to_col)
            successors.append((new_state_tuple, move))

    return successors


def reconstruct_path(node: Node) -> List[Tuple[Tuple[int, ...], ...]]:
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    path.reverse()
    return path

def canonical_state(state: Tuple[Tuple[int, ...], ...]) -> Tuple[Tuple[int, ...], ...]:
    """
    Canonicalizes the state by sorting blocks within each stack and sorting the stacks lexicographically.
    This helps in reducing duplicate states that are permutations of each other.
    """
    sorted_stacks = tuple(sorted(tuple(sorted(stack)) for stack in state if stack))
    return sorted_stacks

def a_star(start_state: Tuple[Tuple[int, ...], ...],
           goal_state: Tuple[Tuple[int, ...], ...],
           blocks: Tuple[int, ...]) -> Optional[List[Tuple[Tuple[int, ...], ...]]]:
    open_set = []
    start_h = heuristic_h1_enhanced(start_state, goal_state, blocks)
    heapq.heappush(open_set, Node(start_state, None, 0, start_h))
    closed_set = set()

    while open_set:
        current_node = heapq.heappop(open_set)

        if current_node.state == goal_state:
            return reconstruct_path(current_node)

        if current_node.state in closed_set:
            continue

        closed_set.add(current_node.state)

        for successor, move in get_successors(current_node.state, blocks):
            canonical_successor = canonical_state(successor)
            if canonical_successor in closed_set:
                continue

            # Avoid reversing the last move
            if current_node.last_move and move == (current_node.last_move[1], current_node.last_move[0]):
                continue

            tentative_cost = current_node.cost + 1  # Each move has a cost of 1
            h = heuristic_h1_enhanced(canonical_successor, goal_state, blocks)
            successor_node = Node(canonical_successor, current_node, tentative_cost, h, move)
            heapq.heappush(open_set, successor_node)

    return None  # No solution found


def ida_star(start_state: Tuple[Tuple[int, ...], ...],
            goal_state: Tuple[Tuple[int, ...], ...],
            blocks: Tuple[int, ...]) -> Optional[List[Tuple[Tuple[int, ...], ...]]]:
    threshold = heuristic_h1_enhanced(start_state, goal_state, blocks)
    path = []
    visited = set()

    def search(node_state, g, threshold, path, last_move):
        f = g + heuristic_h1_enhanced(node_state, goal_state, blocks)
        if f > threshold:
            return f
        if node_state == goal_state:
            path.append(node_state)
            return 'FOUND'
        min_threshold = float('inf')
        for successor, move in get_successors(node_state, blocks):
            canonical_successor = canonical_state(successor)
            if canonical_successor in visited:
                continue
            if last_move and move == (last_move[1], last_move[0]):
                continue
            visited.add(canonical_successor)
            path.append(canonical_successor)
            temp = search(canonical_successor, g + 1, threshold, path, move)
            if temp == 'FOUND':
                return 'FOUND'
            if temp < min_threshold:
                min_threshold = temp
            path.pop()
            visited.remove(canonical_successor)
        return min_threshold

    while True:
        temp = search(start_state, 0, threshold, path, None)
        if temp == 'FOUND':
            return path
        if temp == float('inf'):
            return None
        threshold = temp

def define_goal(blocks: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    # Example Goal: All blocks in one stack sorted from smallest at bottom to largest at top
    sorted_blocks = tuple(sorted(blocks))
    return (sorted_blocks,)

def main():
    # Define the blocks
    blocks = tuple(range(10))  # Blocks 0 through 9

    # Initialize the World with fewer columns
    world = World(num_columns=10)  # Reduced from 10 to 3 for optimization
    print("Initial World State:")
    world.show()

    # Define the Goal State
    goal_state = define_goal(blocks)
    print("Goal State:")
    goal_world = World(grid=[list(col) for col in goal_state], num_columns=10)
    goal_world.show()

    # Convert World grid to tuple of tuples
    start_state = world.grid_as_tuple()

    # Run A* algorithm
    solution = a_star(start_state, goal_state, blocks)

    if solution:
        print("\nSolution found with {} moves:".format(len(solution)-1))
        for step, state in enumerate(solution):
            print(f"Step {step}:")
            temp_world = World(grid=[list(col) for col in state], num_columns=10)
            temp_world.show()
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()
