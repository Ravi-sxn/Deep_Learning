import heapq

# global variable
visited_states = {}  # Maps state to its lowest cost encountered


# My Heuristic
def heuristicFn(state, goal, previous_move=None,barriers=set()):
    x1, y1 = state
    x2, y2 = goal
    manhattan_dist =  abs(x1 - x2)*8 + abs(y1 - y2)*10         # Base heuristic: Weighted Manhattan distance


# Penalize directions that incur high costs
    direction_bias = 0
    if x1 > x2:                                                # Need to move left
        direction_bias += 1                                    # Prefer left as it's cheaper
    if x1 < x2:                                                # Need to move right
        direction_bias += 8                                    # Penalize right as it's expensive
    if y1 > y2:                                                # Need to move up
        direction_bias += 2                                    # Prefer up as it's cheaper
    if y1 < y2:                                                # Need to move down
        direction_bias += 10                                   # Penalize down as it's expensive

    
# Penalty for repeating the same move
    repeat_penalty = 0
    if previous_move:
        if (x1 == x2 and previous_move == 'right') or (y1 == y2 and previous_move == 'down'):
            repeat_penalty = 5                                 # Add penalty for repeating moves that could be suboptimal
   
 # Barrier penalty: If near barriers, discourage expanding nodes here
    barrier_penalty = 0
    for barrier in barriers:
        bx, by = barrier
        if abs(bx - x1) + abs(by - y1) <= 1:                   # Directly adjacent to a barrier
            barrier_penalty += 10 
    return manhattan_dist + direction_bias + repeat_penalty + barrier_penalty




# Fringe list manager (priority queue with heapq)

def addToFringe(fringe, new_entry):
    f_cost, new_cost, neighbor, new_path, direction = new_entry
    
    # Check if the neighbor has already been visited with a lower cost
    if neighbor in visited_states:
        previous_cost = visited_states[neighbor]
        if new_cost < previous_cost:
            # Update the visited state with the new, lower cost
            visited_states[neighbor] = new_cost
            # Add the new entry to the fringe
            heapq.heappush(fringe, new_entry)
    else:
        # Mark the neighbor with its cost in visited_states
        visited_states[neighbor] = new_cost
        # Push the new entry onto the fringe (priority queue)
        heapq.heappush(fringe, new_entry)




# A* function: Only selects the next node to expand based on the fringe
def nextPartialPlanAStar(fringe):
    # Pop the lowest f(n) from the priority queue
    while fringe:
        f_cost, new_cost, state, path, direction = heapq.heappop(fringe)
        # Ensure that only the lowest-cost paths get expanded
        if visited_states[state] == new_cost:
            return f_cost, new_cost, state, path, direction
    return None  # Return None if fringe is empty and no valid paths are found
