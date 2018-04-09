"""
search.py: Search algorithms on grid.
"""
from collections import deque
import heapq

def heuristic(a, b):
    """
    Calculate the heuristic distance between two points.

    For a grid with only up/down/left/right movements, a
    good heuristic is manhattan distance.
    """

    # BEGIN HERE #

    return abs(a[0]-b[0]) + abs(a[1]-b[1])

    # END HERE #


def searchHillClimbing(graph, start, goal):
    """
    Perform hill climbing search on the graph.

    Find the path from start to goal.

    @graph: The graph to search on.
    @start: Start state.
    @goal: Goal state.

    returns: A dictionary which has the information of the path taken.
             Ex. if your path contains and edge from A to B, then in
             that dictionary, d[B] = A. This means that by checking
             d[goal], we obtain the node just before the goal and so on.

             We have called this dictionary, the "came_from" dictionary.
    """

    # Initialise the came_from dictionary
    came_from = {}
    came_from[start] = None

    # BEGIN HERE #
    if start == goal:
        return [start]
        
    stack = [start]
    ext_list = [start]

    while stack:
        node = stack.pop()
        if node == goal:
            break
        neb = graph.neighboursOf(node);
        path = []

        for i in neb:
            if i not in ext_list:
                heapq.heappush(path, (heuristic(i, goal), i))
                ext_list.append(i)
                came_from[i] = node

        path.reverse()
        for i in path:
            stack.append(i[1])

    if goal not in came_from:
        return {}
    temp = goal
    final_path = {}
    while temp != None:
        final_path[temp] = came_from[temp]
        temp = came_from[temp]

    return final_path
    # END HERE #

    return came_from


def searchBestFirst(graph, start, goal):
    """
    Perform best first search on the graph.

    Find the path from start to goal.

    @graph: The graph to search on.
    @start: Start state.
    @goal: Goal state.

    returns: A dictionary which has the information of the path taken.
             Ex. if your path contains and edge from A to B, then in
             that dictionary, d[B] = A. This means that by checking
             d[goal], we obtain the node just before the goal and so on.

             We have called this dictionary, the "came_from" dictionary.
    """


    # Initialise the came_from dictionary
    came_from = {}
    came_from[start] = None


    # BEGIN HERE #
    if start == goal:
        return [start]

    p_queue = [(heuristic(goal, start), start)]
    ext_list = [start]

    while p_queue:
        node = p_queue.pop(0)[1]
        if node == goal:
            break
        neb = graph.neighboursOf(node);
        path = []

        for i in neb:
            if i not in ext_list:
                heapq.heappush(p_queue, (heuristic(i, goal), i))
                ext_list.append(i)
                came_from[i] = node

    if goal not in came_from:
        return {}
    temp = goal
    final_path = {}
    while temp != None:
        final_path[temp] = came_from[temp]
        temp = came_from[temp]

    return final_path
    # END HERE #

    return came_from



def searchBeam(graph, start, goal, beam_length=3):
    """
    Perform beam search on the graph.

    Find the path from start to goal.

    @graph: The graph to search on.
    @start: Start state.
    @goal: Goal state.

    returns: A dictionary which has the information of the path taken.
             Ex. if your path contains and edge from A to B, then in
             that dictionary, d[B] = A. This means that by checking
             d[goal], we obtain the node just before the goal and so on.

             We have called this dictionary, the "came_from" dictionary.
    """

    # Initialise the came_from dictionary
    came_from = {}
    came_from[start] = None

    # BEGIN HERE #
    if start == goal:
        return [start]

    ext_list = [start]
    queue = [start]
    path = []

    while len(queue) != 0:
        node = queue.pop(0)
        if node == goal:
            break

        neb = graph.neighboursOf(node)
        for i in neb:
            if i not in ext_list:
                heapq.heappush(path, (heuristic(goal, i), i))
                ext_list.append(i)
                came_from[i] = node
                # print("Ho")

        length = min(beam_length, len(path))
        for i in range(length):
            queue.append(path[i][1])
            # print("Hi")
        path[:] = []

    if goal not in came_from:
        return {}
    temp = goal
    final_path = {}
    while temp != None:
        final_path[temp] = came_from[temp]
        temp = came_from[temp]

    return final_path
    # END HERE #

    return came_from


def searchAStar(graph, start, goal):
    """
    Perform A* search on the graph.

    Find the path from start to goal.

    @graph: The graph to search on.
    @start: Start state.
    @goal: Goal state.

    returns: A dictionary which has the information of the path taken.
             Ex. if your path contains and edge from A to B, then in
             that dictionary, d[B] = A. This means that by checking
             d[goal], we obtain the node just before the goal and so on.

             We have called this dictionary, the "came_from" dictionary.
    """

    # Initialise the came_from dictionary
    came_from = {}
    came_from[start] = None

    # BEGIN HERE #
    if start == goal:
        return [start]

    p_queue = [(heuristic(goal, start)+0, start)]
    ext_list = [start]
    dist = {start: 0}

    while p_queue:
        node = p_queue.pop(0)[1]
        if node == goal:
            break
        neb = graph.neighboursOf(node);
        path = []

        for i in neb:
            if i not in ext_list:
                dist[i] = dist[node] + 1
                heapq.heappush(p_queue, (heuristic(i, goal)+dist[i], i))
                ext_list.append(i)
                came_from[i] = node

    if goal not in came_from:
        return {}
    temp = goal
    final_path = {}
    while temp != None:
        final_path[temp] = came_from[temp]
        temp = came_from[temp]

    return final_path
    # END HERE #

    return came_from
