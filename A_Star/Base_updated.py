import pygame
from submission_P23CS0006 import *
import random

# Pygame setup
pygame.init()

# Constants for the grid
GRID_SIZE = 9
CELL_SIZE = 50
MARGIN = 5

# Gap between the two grids
GAP_BETWEEN_GRIDS = 100

# Screen size: two grids side by side with a thick gap in between
SCREEN_WIDTH = 2 * (GRID_SIZE * (CELL_SIZE + MARGIN)) + MARGIN + GAP_BETWEEN_GRIDS
SCREEN_HEIGHT = GRID_SIZE * (CELL_SIZE + MARGIN) + MARGIN + 100  # Extra height for displaying stats

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (169, 169, 169)
LIGHT_RED = (255, 102, 102)

# Font for displaying text
FONT = pygame.font.SysFont(None, 36)
FONTsmall = pygame.font.SysFont(None, 18)

# Randomly assign agent to a corner - say 0,0
start_positions = [(0, 0)]
start_state = random.choice(start_positions)

goal_state = (GRID_SIZE-1, GRID_SIZE-1)



# Function to check if a move between two cells is blocked by a barrier
def is_move_blocked(current_state, neighbor, barriers):
    return (current_state, neighbor) in barriers or (neighbor, current_state) in barriers

# Search function that controls the search processdef search(start, goal, barriers):
def search(start, goal, barriers):
    fringe = []
    addToFringe(fringe, (heuristicFn(start, goal), 0, start, [start], None))  
    
    expanded_nodes = 0  # Track the number of expanded nodes (partial plans)
    expanded_set = set()  # Set to track all nodes that have been expanded at least once
    expanded_list_withcost = [] # List to track all nodes and corresponding total cost that have been expanded at least once

    while fringe:
        f_cost, g_cost, current_state, path, prev_direction = nextPartialPlanAStar(fringe)
        
        expanded_nodes += 1  # Each time a node is expanded
        
        
        if(current_state in expanded_set):

            ltcheck = [cost for (st,cost) in expanded_list_withcost if f_cost>=cost and st==current_state]
            if(len(ltcheck)>0):
                print("Error: Same state getting expanded more than once")
                exit()

            for ele in expanded_list_withcost:
                if(ele[0]==current_state):
                    expanded_list_withcost.remove(ele) #removing old entries      
            expanded_list_withcost.append((current_state,f_cost)) #updating the cost to new lower cost
        else:
            expanded_set.add(current_state)
            expanded_list_withcost.append((current_state,f_cost))  # Mark the node as expanded
        
        
        if current_state == goal:
            return path, expanded_nodes, expanded_set, f_cost, g_cost  
        
        neighbors = []
        x, y = current_state
        if x > 0: neighbors.append(((x - 1, y), "up"))
        if x < GRID_SIZE - 1: neighbors.append(((x + 1, y), "down"))
        if y > 0: neighbors.append(((x, y - 1), "left"))
        if y < GRID_SIZE - 1: neighbors.append(((x, y + 1), "right"))
        
        for neighbor, direction in neighbors:
            if is_move_blocked(current_state, neighbor, barriers):
                continue
            
            xn, yn = neighbor
            base_cost = 0
            
            if direction == "down":
                base_cost = 10
            elif direction == "up":
                base_cost = 2
            elif direction == "right":
                base_cost = 8
            elif direction == "left":
                base_cost = 1
            
            # Apply a penalty if the move is in the same direction as the previous move
            penalty = 5 if direction == prev_direction else 0
            
            new_cost = g_cost + base_cost + penalty
            f_cost = new_cost + heuristicFn(neighbor, goal)
            new_path = path + [neighbor]
            addToFringe(fringe, (f_cost, new_cost, neighbor, new_path, direction))
    
    return None, expanded_nodes, expanded_set, None, None  # If no solution found, return expanded nodes count


# Function to generate random barriers
def generate_random_barriers():
    barriers = set()
    
    min_barriers = 3 * GRID_SIZE  
    
    while len(barriers) < min_barriers:
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        if random.choice([True, False]):  # Horizontal or vertical barrier
            if y < GRID_SIZE - 1:  # Horizontal barrier
                barrier = ((x, y), (x, y + 1))
            else:
                continue
        else:
            if x < GRID_SIZE - 1:  # Vertical barrier
                barrier = ((x, y), (x + 1, y))
            else:
                continue
        
        # Ensure no barriers block start or goal positions directly
        if not ((start_state in barrier) or (goal_state in barrier)):
            barriers.add(barrier)
    
    
    return barriers

# Function to draw a grid in Pygame
def draw_grid(screen, offset_x, agent_position, path=[], expanded_nodes_set=set(), barriers=set()):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            color = WHITE
            if (row, col) == agent_position:
                color = BLUE
            elif (row, col) == goal_state:
                color = GREEN
            elif (row, col) in path:
                color = YELLOW
            elif (row, col) in expanded_nodes_set:
                color = LIGHT_RED  # Highlight expanded nodes with a light red color
            
            pygame.draw.rect(
                screen, color,
                [(MARGIN + CELL_SIZE) * col + MARGIN + offset_x, (MARGIN + CELL_SIZE) * row + MARGIN, CELL_SIZE, CELL_SIZE]
            )
            
            

    # Draw barriers
    for barrier in barriers:
        (x1, y1), (x2, y2) = barrier
        if x1 == x2:  # Vertical barrier
            pygame.draw.line(screen, BLACK,
                             [(MARGIN + CELL_SIZE) * y1 + MARGIN + offset_x + CELL_SIZE, (MARGIN + CELL_SIZE) * x1 + MARGIN],
                             [(MARGIN + CELL_SIZE) * y1 + MARGIN + offset_x + CELL_SIZE, (MARGIN + CELL_SIZE) * x1 + MARGIN + CELL_SIZE], 4)
        elif y1 == y2:  # Horizontal barrier
            pygame.draw.line(screen, BLACK,
                             [(MARGIN + CELL_SIZE) * y1 + MARGIN + offset_x, (MARGIN + CELL_SIZE) * x1 + MARGIN + CELL_SIZE],
                             [(MARGIN + CELL_SIZE) * y1 + MARGIN + offset_x + CELL_SIZE, (MARGIN + CELL_SIZE) * x1 + MARGIN + CELL_SIZE], 4)

# Function to render text on the screen
def render_text(screen, text, x, y):
    label = FONT.render(text, True, BLACK)
    screen.blit(label, (x, y))

# Main Pygame loop to display grids
def run_pygame():
    # Set up the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Grid Before and After A* Search")
    
    # Generate random barriers
    barriers = generate_random_barriers()
    
    # Path found by the search, the number of expanded nodes, and the expanded nodes set
    final_path, expanded_nodes, expanded_set, f_cost, g_cost = search(start_state, goal_state, barriers)
    
    # The number of moves in the shortest path
    if(final_path is not None):
        num_moves = len(final_path) - 1 if final_path else 0  # Subtract 1 to exclude the start position

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill(GRAY)
        
        # Draw initial grid (left side)
        draw_grid(screen, 0, start_state, barriers=barriers)
        
        # Draw final grid with the solution path and expanded nodes (right side with a thick gap in between)
        draw_grid(screen, SCREEN_WIDTH // 2 + GAP_BETWEEN_GRIDS // 2, start_state, final_path, expanded_set, barriers)
        if(final_path is not None):
            # Display the number of moves and expanded nodes
            render_text(screen, f"Shortest path found: Moves: {num_moves}, Goal State - FCost: {f_cost}, GCost: {g_cost}", 10, SCREEN_HEIGHT - 90)
            render_text(screen, f"Partial plans expanded: {expanded_nodes}", 10, SCREEN_HEIGHT - 50)
        else:
            render_text(screen, f"No solution found", 10, SCREEN_HEIGHT - 90)
        
        pygame.display.flip()
    
    pygame.quit()

# Run the Pygame visualization
run_pygame()