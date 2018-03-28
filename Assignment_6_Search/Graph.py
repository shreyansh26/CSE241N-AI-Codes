''' class Graph defines the bfs and dfs functions that operate on the graph  '''

class Graph:

    def __init__(self, adj):
        n = len(adj)
        self.graph = dict()
        for i in range(n):
            temp = []
            for j in range(n):
                if adj[i][j]:
                    temp.append(j)
            self.graph[i] = set(temp)

    def bfs_paths(self, start, goal):
        '''
        Generate and return any path from start to goal using breadth-first search
        Input : start node, goal node
        Output : list of nodes from to be traversed to reach from start to goal(the first node in this list will be the start node and the last node will be the goal node)
        '''
        #BEGIN YOUR CODE HERE
        visited = [False]*len(self.graph)
        queue = []
        prev = {}

        visited[start] = True
        queue.append(start)

        while len(queue) != 0:
            s = queue.pop(0)

            for i in self.graph[s]:
                if not visited[i]:
                    visited[i] = True
                    prev[i] = s
                    queue.append(i)

        x = goal
        path = []
        while x!=start:
            path.append(x)
            x = prev[x]

        path.append(x)
        path = path[::-1]
        # print(path)
        return path
        #END YOUR CODE HERE

    def dfs_paths(self, start, goal):
        '''
        Generate and return any path from start to goal using depth-first search
        Input : start node, goal node
        Output : list of nodes from to be traversed to reach from start to goal(the first node in this list will be the start node and the last node will be the goal node)
        '''
        #BEGIN YOUR CODE HERE
        # print(self.graph)
        prev = {}
        visited = [False]*len(self.graph)
        stack = []

        stack.append(start)

        while len(stack) != 0:
            s = stack.pop()
            if not visited[s]:
                visited[s] = True;

            for i in self.graph[s]:
                if not visited[i]:
                    prev[i] = s
                    stack.append(i)

        # print(prev)
        x = goal
        path = []
        while x!=start:
            path.append(x)
            x = prev[x]

        path.append(x)
        path = path[::-1]
        # print(path)
        return path
    #END YOUR CODE HERE
