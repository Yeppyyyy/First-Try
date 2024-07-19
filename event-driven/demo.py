import heapq
import numpy as np

# 定义优先队列类
class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def pop(self):
        return heapq.heappop(self.elements)[1]

    def remove(self, item):
        self.elements = [(pri, it) for pri, it in self.elements if it != item]
        heapq.heapify(self.elements)

    def top_key(self):
        return self.elements[0][0] if self.elements else float('inf')

# 定义 D* Lite 算法类
class DStarLite:
    def __init__(self, adj_matrix, start, goal):
        self.adj_matrix = adj_matrix
        self.start = start
        self.goal = goal
        self.num_nodes = adj_matrix.shape[0]
        self.g = [float('inf')] * self.num_nodes
        self.rhs = [float('inf')] * self.num_nodes
        self.U = PriorityQueue()
        self.km = 0
        self.initialize()

    def initialize(self):
        self.rhs[self.goal] = 0
        self.U.put(self.goal, self.calculate_key(self.goal))

    def h(self, a, b):
        # 曼哈顿距离作为启发函数（假设节点为一维索引）
        return abs(a - b)

    def update_vertex(self, u):
        if u != self.goal:
            self.rhs[u] = min(self.adj_matrix[u, v] + self.g[v] for v in range(self.num_nodes) if self.adj_matrix[u, v] != float('inf'))
        self.U.remove(u)
        if self.g[u] != self.rhs[u]:
            self.U.put(u, self.calculate_key(u))

    def compute_shortest_path(self):
        while self.U.top_key() < self.calculate_key(self.start) or self.rhs[self.start] != self.g[self.start]:
            k_old = self.U.top_key()
            u = self.U.pop()
            if k_old < self.calculate_key(u):
                self.U.put(u, self.calculate_key(u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in range(self.num_nodes):
                    if self.adj_matrix[u, s] != float('inf'):
                        self.update_vertex(s)
            else:
                self.g[u] = float('inf')
                for s in range(self.num_nodes):
                    if self.adj_matrix[u, s] != float('inf'):
                        self.update_vertex(s)
                self.update_vertex(u)

    def calculate_key(self, s):
        return min(self.g[s], self.rhs[s]) + self.h(self.start, s) + self.km

    def update_edge_cost(self, u, v, new_cost):
        self.adj_matrix[u, v] = new_cost
        self.update_vertex(u)
        self.compute_shortest_path()

    def get_path(self):
        path = []
        current = self.start
        while current != self.goal:
            path.append(current)
            next_node = min(
                (v for v in range(self.num_nodes) if self.adj_matrix[current, v] != float('inf')),
                key=lambda v: self.g[v] + self.adj_matrix[current, v]
            )
            current = next_node
        path.append(self.goal)
        return path

# 示例邻接矩阵
adj_matrix = np.array([
    [0, 5, float('inf'), 10],
    [float('inf'), 0, 3, float('inf')],
    [float('inf'), float('inf'), 0, 1],
    [float('inf'), float('inf'), float('inf'), 0]
])

# 定义起点和终点
start, goal = 0, 3

# 初始化 D* Lite 算法
dstar = DStarLite(adj_matrix, start, goal)

# 计算初始路径
dstar.compute_shortest_path()
print("Initial Path:")
print(dstar.get_path())

# 假设在某个时刻更新图中的某些边的权重
dstar.update_edge_cost(1, 2, 2)
dstar.update_edge_cost(0, 1, 6)

# 更新路径
print("Updated Path:")
print(dstar.get_path())
