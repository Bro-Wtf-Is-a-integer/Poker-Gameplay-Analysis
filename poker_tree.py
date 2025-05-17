from contextlib import nullcontext

from poker_game_example import *
from collections import deque

INIT_AGENT_STACK = 100
MAX_HANDS = 5

class TreeNode:
    def __init__(self, id = None, decision=None, parent=None, children=None, result=None, stack=None, hand_strength=None,
                 player_action=None, game_state=None, acting_agent=None, current_hand=None):  # Add current_hand
        self.id = id
        self.decision = decision
        self.parent = parent  # Reference to the parent node
        self.children = children if children is not None else []
        self.result = result  # Result at the leaf node (e.g., 'Win', 'Lose', 'Draw')
        self.stack = stack
        self.hand_strength = hand_strength  # The strength of the player's hand (optional)
        self.player_action = player_action  # The action the player has taken (optional)
        self.game_state = game_state  # The game state at this node
        self.acting_agent = acting_agent  # Store whether it's the agent's or opponent's turn
        self.current_hand = current_hand  # Set current_hand




    def __repr__(self):
        phase = self.game_state.phase if self.game_state else 'N/A'
        acting_agent = self.acting_agent if self.acting_agent else 'N/A'
        return (f"TreeNode(id = {self.id}, decision={self.decision}, result={self.result}, hand_strength={self.hand_strength}, "
                f"phase={phase}, acting_agent={acting_agent}, current_hand={self.current_hand}), stack = {self.stack})")


    def add_child(self, child_node):
        self.children.append(child_node)

    def getStack(self):
        return self.stack




def is_agent_turn(game_state):
    if game_state.acting_agent is None:
        raise ValueError("The acting agent has not been set!")
    return game_state.acting_agent == 'agent'

def goal_check(state):
    """
    Checks if the given state satisfies the goal conditions.

    Args:
        state: A GameState object representing the current game state.

    Returns:
        True if the goal condition is met, False otherwise.
    """
    agent_stack = state.agent.stack
    opponent_stack = state.opponent.stack
    current_hand = state.nn_current_hand

    # Goal state conditions
    if agent_stack == 0 or opponent_stack == 0:  # One of the players is out of money
        return True

    if current_hand == MAX_HANDS:  # Maximum number of hands reached
        return True

    return False

def heuristic(node):
    remaining_hands = MAX_HANDS - node.game_state.nn_current_hand
    stack_diff = (INIT_AGENT_STACK + 100) - node.game_state.agent.stack #goal stack, the smaller the better

    # safeguard mechanism as well as debugging it, could be due to initialization
    current_hand_strength = node.game_state.agent.current_hand_strength
    if not isinstance(current_hand_strength, (int, float)):
        print("fall back")
        current_hand_strength = 1

    # desired
    if stack_diff == 0:
        return 0

    #safeguard
    if remaining_hands == 0:
        return float('inf')

    # Estimate potential gain per remaining hand based on hand strength
    gains = current_hand_strength * remaining_hands

    # The heuristic
    return stack_diff / gains


def buildTree(init_state):
    root = TreeNode(decision=None, parent=None, game_state=init_state, stack=init_state.agent.stack, current_hand=init_state.nn_current_hand)
    states_queue = deque([(init_state, root)])  # Ensure deque is initialized properly
    node = 0

    while states_queue:
        current_state, current_node = states_queue.popleft()
        node += 1

        if goal_check(current_state):
            continue


        next_states = get_next_states(current_state)

        for next_state in next_states:
            if is_agent_turn(next_state):
                player_action = next_state.agent.action
                hand_strength = next_state.agent.current_hand_strength
                stack = next_state.agent.stack
            else:
                player_action = next_state.opponent.action
                hand_strength = next_state.opponent.current_hand_strength
                stack = next_state.opponent.stack

            result = next_state.showdown_info if next_state.phase == 'SHOWDOWN' else None

            # Ensure current_hand is updated properly for each new node
            current_hand = next_state.nn_current_hand

            new_node = TreeNode(
                id = node,
                decision=player_action,
                parent=current_node,
                game_state=next_state,
                stack=stack,
                hand_strength=hand_strength,
                acting_agent='agent' if is_agent_turn(next_state) else 'opponent',  # Set acting_agent
                result=result,
                current_hand=current_hand,  # Pass current_hand to each new node

            )


            current_node.add_child(new_node)
            states_queue.append((next_state, new_node))

    return root, node


def print_tree(node, depth=0):
    print("  " * depth + repr(node))  # This will now include acting_agent and current_hand
    for child in node.children:
        print_tree(child, depth + 1)


def bfs_(root):

    queue = deque([root])  # Queue to hold nodes to explore
    visited = set()  # Keep track of visited nodes
    paths = {root: []}  # Dictionary to store the path to each node
    nodes = 0  # Count the total number of nodes processed
    biddings = 0

    while queue:
        current_node = queue.popleft()  # Get the next node to process
        current_state = current_node.game_state  # Extract the game state
        nodes += 1  # Increment the nodes count

        # Skip already visited nodes
        if current_node in visited:
            continue

        if current_state.phase == 'BIDDING':
            biddings += 1
        # Goal condition (agent has enough stack)
        if current_state.agent.stack >= INIT_AGENT_STACK + 100:
            print(f"Goal reached at hand {current_state.nn_current_hand}")
            # Return the path from the start node to the current node
            return paths[current_node], nodes, len(paths[current_node]), biddings,current_state.nn_current_hand

        visited.add(current_node)

        # Explore all child nodes and add them to the queue
        for child_node in current_node.children:
            if child_node not in visited:
                queue.append(child_node)
                # Update the path to the child node (add the current node to the path)
                paths[child_node] = paths[current_node] + [child_node]

    print("Goal not reached.")
    return None, nodes, len(visited)  # If no goal is found



def dfs_(root):
    stack = [root]  # Stack to hold nodes to explore
    visited = set()  # Keep track of visited nodes
    paths = {root: []}  # Dictionary to store the path to each node
    nodes = 0  # Count the total number of nodes processed
    biddings = 0

    while stack:
        current_node = stack.pop()  # Get the next node to process
        current_state = current_node.game_state  # Extract the game state
        nodes += 1  # Increment nodes count

        # Skip already visited nodes
        if current_node in visited:
            continue

        if current_state.phase == 'BIDDING':
            biddings += 1

        # Goal condition (agent has enough stack)
        if current_state.agent.stack >= INIT_AGENT_STACK + 100:
            print(f"Goal reached at hand {current_state.nn_current_hand}")
            # Return the path from the start node to the current node
            return paths[current_node], nodes, len(paths[current_node]), biddings,current_state.nn_current_hand

        visited.add(current_node)

        # Explore all child nodes and add them to the stack
        for child_node in current_node.children:
            if child_node not in visited:
                stack.append(child_node)
                # Update the path to the child node (add the current node to the path)
                paths[child_node] = paths[current_node] + [child_node]

    print("Goal not reached.")
    return None, nodes, len(visited)  # If no goal is found



import heapq
def gs_(root):
    import heapq  # For priority queue
    open_list = []  # Priority queue for nodes to explore
    total_nodes_expanded = 0  # Counter for expanded nodes
    visited_nodes = set()  # Set to track visited nodes
    entry_counter = 0  # Counter for tie-breaking in the priority queue
    path_tracker = {root: []}  # Tracks paths from the root to each node

    # Initialize heuristic value for the root
    root_heuristic = heuristic(root)
    heapq.heappush(open_list, (root_heuristic, entry_counter, root))  # Push root to the priority queue
    total_biddings = 0  # Counter for bidding phases encountered

    while open_list:
        # Get the node with the lowest heuristic value
        _, _, current_node = heapq.heappop(open_list)
        total_nodes_expanded += 1  # Increment the node expansion counter

        # Skip if the node has already been visited
        if current_node in visited_nodes:
            continue

        # If the node is in the bidding phase, count it
        if current_node.game_state.phase == 'BIDDING':
            total_biddings += 1

        # Check if the goal condition is met
        current_state = current_node.game_state
        if current_state.agent.stack >= INIT_AGENT_STACK + 100:
            return (
                path_tracker[current_node],  # Path to the goal
                total_nodes_expanded,  # Total nodes expanded
                len(path_tracker[current_node]),  # Path length
                total_biddings,  # Total bidding phases
                current_state.nn_current_hand  # Current hand at the goal
            )

        # Mark the node as visited
        visited_nodes.add(current_node)

        # Explore the children of the current node
        for child in current_node.children:
            if child not in visited_nodes:
                child_heuristic = heuristic(child)  # Calculate heuristic for the child
                entry_counter += 1  # Increment the tie-breaker counter
                heapq.heappush(open_list, (child_heuristic, entry_counter, child))  # Add to priority queue
                # Update path tracker
                path_tracker[child] = path_tracker[current_node] + [child]

    # Return None if no goal is found
    return None, total_nodes_expanded


def sort_and_print_tree_nodes(tree_nodes, sort_by='id'):
    """
    Sort and format the TreeNode objects for easier printing.

    Args:
        tree_nodes: A list of TreeNode objects.
        sort_by: The attribute by which to sort the nodes. Options: 'cost', 'stack', 'hand_strength'.
    """
    # Sort the tree_nodes based on the specified attribute (default is 'cost')
    if sort_by == 'id':
        tree_nodes.sort(key=lambda node: node.id)
    elif sort_by == 'stack':
        tree_nodes.sort(key=lambda node: node.stack)
    elif sort_by == 'hand_strength':
        tree_nodes.sort(key=lambda node: node.hand_strength)

    # Print the sorted nodes in a readable format
    for node in tree_nodes:
        print(f"id: {node.id}")
        print(f"Decision: {node.decision}")
        print(f"Result: {node.result}")
        print(f"Hand Strength: {node.hand_strength}")
        print(f"Acting Agent: {node.acting_agent}")
        print(f"Current Hand: {node.current_hand}")
        print(f"Stack: {node.stack}")
        print("-" * 50)



agent = PokerPlayer(current_hand_=None, stack_=INIT_AGENT_STACK, action_=None, action_value_=None)
opponent = PokerPlayer(current_hand_=None, stack_=INIT_AGENT_STACK, action_=None, action_value_=None)


init_state = GameState(
    nn_current_hand_=0,
    nn_current_bidding_=0,
    phase_='INIT_DEALING',
    pot_=0,
    acting_agent_= None,
    agent_=agent,
    opponent_=opponent
)
root, nodes = buildTree(init_state)


bfs_states, bfs_nodes, bfs_visited, bfs_biddings,bfs_hands = bfs_(root)
dfs_states, dfs_nodes, dfs_visited, dfs_biddings,dfs_hands = dfs_(root)
gs_states, gs_nodes, gs_visited, gs_biddings, gs_hands = gs_(root)
print(f"Tree node {nodes}")
print(f"bfs nodes: {bfs_nodes} and bfs visited: {bfs_visited} and bfs biddings: {bfs_biddings} and current hand {bfs_hands}")
print(f"dfs nodes: {dfs_nodes} and dfs visited: {dfs_visited} and dfs biddings: {dfs_biddings} and current hand {dfs_hands}")
print(f"gs nodes: {gs_nodes} and gs visited: {gs_visited} and gs biddings: {gs_biddings} and current hand {gs_hands}")





