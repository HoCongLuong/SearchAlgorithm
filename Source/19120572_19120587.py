import os
import matplotlib.pyplot as plt
from collections import defaultdict
from queue import PriorityQueue
from numpy import sqrt
from copy import copy
import time
from pathlib import Path

def visualize_maze(matrix, bonus, start, end, routes = None, names = None, times = None):
    """
    Args:
      1. matrix: The matrix read from the input file,
      2. bonus: The array of bonus points,
      3. start, end: The starting and ending points,
      4. route: The route from the starting point to the ending one, defined by an array of (x, y), e.g. route = [(1, 2), (1, 3), (1, 4)]
    """
    # 1. Define walls and array of direction based on the route
    walls = [(i, j) for i in range(len(matrix))
             for j in range(len(matrix[0])) if matrix[i][j] == 'x']

    for r in range(len(routes)):
        if routes:
            direction = []
            for i in range(2, len(routes[r])):
                if routes[r][i][0]-routes[r][i-1][0] > 0:
                    direction.append('v')  # ^
                elif routes[r][i][0]-routes[r][i-1][0] < 0:
                    direction.append('^')  # v
                elif routes[r][i][1]-routes[r][i-1][1] > 0:
                    direction.append('>')
                else:
                    direction.append('<')

        # 2. Drawing the map
        ax = plt.figure(dpi=100).add_subplot(111)

        for i in ['top', 'bottom', 'right', 'left']:
            ax.spines[i].set_visible(False)

        plt.scatter([i[1] for i in walls], [-i[0] for i in walls],
                    marker='X', s=100, color='black')

        plt.scatter([i[1] for i in bonus], [-i[0] for i in bonus],
                    marker='P', s=100, color='green')

        plt.scatter(start[1], -start[0], marker='*',
                    s=100, color='gold')

        if routes:
            for i in range(len(routes[r])-2):
                plt.scatter(routes[r][i+1][1], -routes[r][i+1][0],
                            marker=direction[i], color='silver')

        plt.text(end[1], -end[0], 'EXIT', color='red',
                horizontalalignment='center',
                verticalalignment='center')
        plt.xticks([])
        plt.yticks([])
        bonus_points_cost = 0
        for index in routes[r]:
            if(bonus[index] != []):
                bonus_points_cost = bonus_points_cost + bonus[index]
            else:
                bonus_points.pop(index) 
        title = "Cost: " + str(len(routes[r]) + bonus_points_cost)
        if(names):
            title = names[r] + " - " + title
        if(times):
            title = title + " - Time: " + str(times[r]) 
        plt.title(title)
    plt.show()

    print(f'Starting point (x, y) = {start[0], start[1]}')
    print(f'Ending point (x, y) = {end[0], end[1]}')

    for point in bonus:
        print(point[0], point[1], bonus[point],)

    for _, point in enumerate(bonus):
        print(
            f'Bonus point at position (x, y) = {point[0], point[1]} with point {point[2]}')


def read_file(file_name: str = 'maze.txt'):
    f = open(file_name, 'r')
    n_bonus_points = int(next(f))
    bonus_points = defaultdict(list)
    for i in range(n_bonus_points):
        x, y, reward = map(int, next(f)[:-1].split(' '))
        bonus_points[x,y] = reward

    text = f.read()
    matrix = [list(i) for i in text.splitlines()]
    f.close()

    return bonus_points, matrix


def detached_matrix(matrix):

    graph = defaultdict(list)

    for i in range(1,len(matrix) -1):
        for j in range(1,len(matrix[0]) - 1):
            if matrix[i][j] != 'X':
                # down
                if(matrix[i + 1][j] == ' '):
                    graph[i, j].append((i + 1, j))
                # right
                if(matrix[i][j + 1] == ' '):
                    graph[i, j].append((i, j + 1))
                # left
                if(matrix[i][j - 1] == ' '):
                    graph[i, j].append((i, j - 1))
                # up
                if(matrix[i - 1][j] == ' '):
                    graph[i, j].append((i - 1, j))

    return graph


def BFS(graph, start, goal):

    """
        graph: include vertex: desVertex1,....desVertexN

        return shortest path of start and goal
    """

    # create visited dict to mark coord when we move
    visited = defaultdict(lambda: False)

    # create queue
    queue = []

    # result
    result = defaultdict(list)

    queue.append(start)
    visited[start] = True

    while queue:
        # Dequeue a vertex from
        # queue and print it
        s = queue.pop(0)

        for i in graph[s]:
            if visited[i] == False:
                queue.append(i)
                visited[i] = True
                result[i] = s
                if i == goal:
                    return result


    return result


def DFS(graph, start, goal):
    """
        graph: include vertex: desVertex1,....desVertexN

        return shortest path of start and goal
    """

    # create a visited dict
    visited = defaultdict(lambda: False)

    # path for looking shortest path
    result = defaultdict(list)

    # stack for dfs
    startStack = []
    startStack.append(start)

    while True:
        # get last element
        vertex = startStack.pop()
       
        visited[vertex] = True

        for _adj in graph[vertex]:
            if visited[_adj] == False:
                startStack.append(_adj)
                visited[_adj] = True
                result[_adj] = vertex

                if(_adj == goal):
                    return result


def GBFS(graph, start, goal, heuristic):
    """
        graph: include vertex: desVertex1,....desVertexN

        return shortest path of start and goal
    """

    # create visited dict to mark coord when we move
    visited = defaultdict(lambda: False)

    # create queue
    queue = PriorityQueue()

    # result
    result = defaultdict(list)

    queue.put((0,start))
    visited[start] = True

    while queue:
        # Dequeue a vertex from
        # queue and print it
        s = queue.get()[1]
        if s == goal:
            return result

        for i in graph[s]:
            if visited[i] == False:
                queue.put((heuristic(i,goal),i))

                # make visited for this vertex
                visited[i] = True

                # previous vertex is S
                result[i] = s

    return result


def AS(graph, start, goal, heuristic):
    """
        graph: include vertex: desVertex1,....desVertexN

        return shortest path of start and goal
    """

    # create visited dict to mark coord when we move
    visited = defaultdict(lambda: False)

    # create queue
    queue = PriorityQueue()

    # result
    result = defaultdict(list)

    # f
    f = defaultdict(lambda: 0)

    queue.put((0,start))
    visited[start] = True

    while queue:
        # Dequeue a vertex from
        # queue and print it
        s = queue.get()[1]
        
        if s == goal:
            return result
    
        for i in graph[s]:
            if visited[i] == False:
                # 4 điểm lận cận tốn chi phí 
                # là 1 từ điểm bắt đầu
                f[i] = f[s] + 1

                queue.put((heuristic(i,goal) 
                + f[i], i))

                # make visited for this vertex
                visited[i] = True

                # previous vertex is S
                result[i] = s

    return result


def MySearch(graph, start, goal, heuristic, _bonus_points = None):
    """
        graph: include vertex: desVertex1,....desVertexN

        return shortest path of start and goal
    """
    index = 0

    bonus_points = copy(_bonus_points)

    chi_phi = 1
    max_bonus_point = 0
    for i in bonus_points:
        if max_bonus_point < abs(bonus_points[i]): 
            max_bonus_point = abs(bonus_points[i])

    temp = sqrt(max_bonus_point)/chi_phi
    chi_phi = chi_phi * temp
    for i in bonus_points:
        bonus_points[i] = bonus_points[i]*temp

    
    # copy another bonus point    
    bonus_points_temp = copy(bonus_points)

    # create visited dict to mark coord when we move
    visited = defaultdict(lambda: False)

    # create queue to contain vertex
    queue = PriorityQueue()

    # queue to contain bonus point
    queue_bonus_points = []

    # result
    result = defaultdict(list)

    # temp result
    _result = defaultdict(list)

    # f la chi phi cua duong di
    f = defaultdict(lambda: 0)
    # f temp
    _f = defaultdict(lambda: 0)

    queue.put((0,start))
    visited[start] = True

    while queue:
        # Dequeue a vertex from
        # queue and print it
        s = queue.get()[1]

        if s == goal:
            if(queue_bonus_points != []):
                temp = queue_bonus_points.pop(0)
                while temp:
                    result[temp] = _result[temp]
                    temp = _result[temp]

                    for vertex in _result:
                                    if(result[vertex] == []):
                                        result[vertex] = _result[vertex]
            return result, index
        
        for i in graph[s]:
            _f[i] = _f[s] + chi_phi
            if(bonus_points):
                if(bonus_points[i] and result[s] != i): 
                    _f[i] = _f[s] + bonus_points[i]
                elif (bonus_points[i] == []):
                        bonus_points.pop(i)
            if (visited[i] == False or _f[i] < f[i]):
                # diem nao ma no la cha cua bonus
                if [bonus for bonus in bonus_points_temp if result[bonus] == i] != []:
                    for _bonus in bonus_points_temp:
                        if(result[_bonus] == []): 
                            result.pop(_bonus)
                    
                    continue    
                # 4 điểm lận cận tốn chi phí 
                # là 1 từ điểm bắt đầu
                f[i] = _f[i]
                if(bonus_points):
                    if(bonus_points[i] and result[s] != i):
                        
                        result[i] = s

                        queue_bonus_points.append(i)
                        
                        if(len(queue_bonus_points) == 2):
                            temp = queue_bonus_points.pop(0)
                            while temp:
                                result[temp] = _result[temp]
                                temp = _result[temp]

                            for vertex in _result:
                                if(result[vertex] == []):
                                    result[vertex] = _result[vertex]

                        bonus_points.pop(i)
                        
                        _result = copy(result)

                        result.clear()
                    elif (bonus_points[i] == []):
                        bonus_points.pop(i)

                queue.put((heuristic(i,goal) + f[i], i))

                # make visited for this vertex
                visited[i] = True

                # previous vertex is S
                result[i] = s

                index +=1

    return result, index


def path(source, start, goal):
    
    """ 
        source: dict(start: des)
        start: point start
        gold: exit 
    """

    path = [goal]
    _temp = goal

    i = 0

    while _temp:
        i+=1
        temp = source[_temp]
        _temp = temp
        path.append(temp)
        if(_temp == start or i > 1000):
            return path
    
    return path


def space(vertex, goal):
    return sqrt((vertex[0] - goal[0])**2 + (vertex[1] - goal[1])**2)


def space2(vertex, goal):
    return abs(goal[0] - vertex[0]) + abs(goal[1] - vertex[1])

print("Moi nhap ten file: ", end = "")
fileName = str(input())


try: 

    try:
        bonus_points, matrix = read_file(f'KiemThu\{fileName}')
    except:
        bonus_points, matrix = read_file(f'..\KiemThu\{fileName}')
    graph = detached_matrix(matrix)

    print(f'The height of the matrix: {len(matrix)}')
    print(f'The width of the matrix: {len(matrix[0])}')

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 'S':
                start = (i, j)

            elif matrix[i][j] == ' ':
                if (i == 0) or (i == len(matrix)-1) or (j == 0) or (j == len(matrix[0])-1):
                    end = (i, j)

            else:
                pass

    bonus_points_temp = copy(bonus_points)
    startMySearch = time.time()
    s, index = MySearch(graph, start, end, space, bonus_points_temp)
    endMySearch  = time.time()
    _path = path(s,start,end)
    _path.reverse()
    
    startASearch = time.time()
    aSearch = AS(graph, start, end, space)
    endASearch  = time.time()
    _aSearch = path(aSearch,start,end)
    _aSearch.reverse()


    startGBFS = time.time()
    gbfs = GBFS(graph, start, end, space)
    endGBFS  = time.time()
    _gbfs = path(gbfs,start,end)
    _gbfs.reverse()
    
    startBFS = time.time()
    bfs = BFS(graph,start,end)
    endBFS  = time.time()
    _bfs = path(bfs,start,end)
    _bfs.reverse()
    
    startDFS = time.time()
    dfs = DFS(graph,start,end)
    endDFS  = time.time()
    _dfs = path(dfs,start,end)
    _dfs.reverse()
    
    visualize_maze(matrix, bonus_points, start, end,
    [_path, _aSearch, _gbfs, _bfs, _dfs,],
    ["My Search", "A Search", "Greedy Best First Search", "Breadth First Search", "Depth First Search"],
    [endMySearch - startMySearch, endASearch - startASearch, endGBFS - startGBFS, endBFS - startBFS, endDFS - startDFS],)

except:
    print("An exception occurred")
