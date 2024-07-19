import data_gen as gen
import uuid
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from queue import PriorityQueue

parameters = {
    "station_num": 25,
    "center_num": 5,
    "packet_num": 2000,
}

log_events = ["ARRIVED", "PROCESSING", "SENT"]

# Location
class Location:
    def __init__(self, ID, pos, throughput, delay, cost):
        self.ID = ID
        self.pos = pos
        self.throughput = throughput
        self.delay = delay
        self.cost = cost
        self.buffer = {'arrived':[], 'processing': [], 'ready_to_send':[]}
        self.pack_passby = {}
        self.capa = [throughput, 0]
        self.waiting = []

    def sort_pack(self):
        pack_passby = {k: v for k, v in sorted(self.pack_passby.items(), key=lambda item: item[1])}
        return pack_passby

    # 更新timetick和处理量
    def update(self, t):
        if self.capa[1] < int(t):
            self.capa[1] = int(t)
            self.capa[0] = self.throughput


# Route
class Route:
    def __init__(self, src, dst, time, cost):
        self.src = src
        self.dst = dst
        self.time = time
        self.cost = cost

# Package
class Package:
    def __init__(self, ID, time_created, src, dst, category):
        self.ID = ID
        self.time_created = time_created
        self.src = src
        self.dst = dst
        self.category = category
        self.tracking_info = []

# simulator
# 存模拟的数据 函数等
class simulator:
    def __init__(self, data):
        # 导入模拟的初始数据（input）
        self.locations = self.setlocation(data)
        self.routes = self.setroutes(data)
        self.packages = self.setpackages(data)
        self.G = self.route_graph(data) # 把route转换成graph，用去年图论的nx包
        self.tracking_info = {}
        self.on_route = []  # 在路上的包裹
        self.route_time_matrix = self.matrix("time")  # 路线的时间矩阵
        self.route_cost_matrix = self.matrix("cost")  # 路线的成本矩阵
        self.route_min_cost = self.chemin_cost_matrix()   # 最小成本路线矩阵
        self.route_min_time = self.chemin_time_matrix()
        self.events = self.init_events()
        self.waiting = []

    # 下面这几个set都是用来初始化用的
    def setlocation(self, data):
        locations = []
        for ID, pos, prop in zip(range(gen.parameters["station_num"]),
                                 data["station_pos"],
                                 data["station_prop"]):
            loc = Location("s" + str(ID), pos, prop[0], prop[1], prop[2])
            locations.append(loc)
        for ID, pos, prop in zip(range(gen.parameters["center_num"]),
                                 data["center_pos"],
                                 data["center_prop"]):
            loc = Location("c" + str(ID), pos, prop[0], prop[1], prop[2])
            locations.append(loc)
        return locations

    def setroutes(self, data):
        routes = []
        for src, dst, time, cost in data["edges"]:
            route = Route(src, dst, time, cost)
            routes.append(route)
        return routes

    def setpackages(self, data):
        packages = []
        for pkg_data in data["packets"]:
            pkg_id = pkg_data[0]
            pkg = Package(pkg_data[0], pkg_data[1], pkg_data[2], pkg_data[3], pkg_data[4])
            packages.append(pkg)
        return packages

    # 把route数据转换成图
    def route_graph(self, routes):
        G = nx.DiGraph()
        for route in self.routes:
            G.add_edge(route.src, route.dst, time=route.time, cost=route.cost)
        return G

     # 返回一个邻接矩阵（表示两个location之间的time 或者 cost）
    def matrix(self, time_cost):
        matrix = np.full((parameters["station_num"] + parameters["center_num"],
                        parameters["station_num"] + parameters["center_num"]), np.inf)
        np.fill_diagonal(matrix, 0)
        # time矩阵
        if time_cost == 'time':
            for route in self.routes:
                if route.src[0] == 's':
                    if route.dst[0] == 's':
                        matrix[int(route.src[1:])][int(route.dst[1:])] = route.time
                    else:
                        matrix[int(route.src[1:])][int(route.dst[1:]) + parameters["station_num"]] = route.time
                else:
                    if route.dst[0] == 's':
                        matrix[int(route.src[1:]) + parameters["station_num"]][int(route.dst[1:])] = route.time
                    else:
                        matrix[int(route.src[1:]) + parameters["station_num"]][
                            int(route.dst[1:]) + parameters["station_num"]] = route.time
            # cost矩阵
        else:
            for route in self.routes:
                if route.src[0] == 's':
                    if route.dst[0] == 's':
                        matrix[int(route.src[1:])][int(route.dst[1:])] = route.cost
                    else:
                        matrix[int(route.src[1:])][int(route.dst[1:]) + parameters["station_num"]] = route.cost
                else:
                    if route.dst[0] == 's':
                        matrix[int(route.src[1:]) + parameters["station_num"]][int(route.dst[1:])] = route.cost
                    else:
                        matrix[int(route.src[1:]) + parameters["station_num"]][
                            int(route.dst[1:]) + parameters["station_num"]] = route.cost
        return matrix

    # 返回一个时间最短路径矩阵，比如第二行第五列代表s2到s5的最短路径
    def chemin_time_matrix(self):
        time_matrix = self.route_time_matrix
        loc_delay = [loc.delay for loc in self.locations]
        chemin_time = []
        for src in self.locations:
            src = int(src.ID[1:])
            chemin = self.dijkstra(time_matrix, loc_delay, src)
            chemin_time.append(chemin)
        return chemin_time

    # 同理，cost最少的路径
    def chemin_cost_matrix(self):
        cost_matrix = self.route_cost_matrix
        loc_cost = [loc.cost for loc in self.locations]
        chemin_cost = []
        for src in self.locations:
            src = int(src.ID[1:])
            chemin = self.dijkstra(cost_matrix, loc_cost, src)
            chemin_cost.append(chemin)
        return chemin_cost

    # 初始化事件池
    def init_events(self):
        events = []
        for pack in self.packages:
            events.append([pack, pack.time_created, pack.src, "Created"])
        events.sort(key=lambda x: x[1])
        return events
    # ------------------初始化函数到这结束------------------#

    # 最短路径dijkstra算法，只返回路径不返回最短成本 （看seb的课
    def dijkstra(self, M, loc_M, s0):
        sk = s0
        n = len(M)
        delta = [np.inf] * n
        delta[s0] = 0
        mark = [False] * n
        mark[s0] = True
        chemin = [[]] * n
        chemin[s0] = [s0]
        while sum(mark) < n:
            for t in range(n):
                if M[sk][t] != 0 and delta[sk] + M[sk][t] + loc_M[t] < delta[t]:
                    delta[t] = delta[sk] + M[sk][t] + loc_M[t]
                    chemin[t] = chemin[sk] + [t]
            delta_min = np.inf
            for s in range(n):
                if not (mark[s]) and delta[s] < delta_min:
                    sk = s
                    delta_min = delta[sk]
            mark[sk] = True
        return chemin


    # 用nx包画站和路径的图
    def draw_route(self):

        # 时间图
        plt.figure()
        pos = nx.spring_layout(self.G)
        for node in self.G.nodes:
            if node[0] == 'c':
                nx.draw_networkx_nodes(self.G, pos, nodelist=[node], node_color='grey', node_size=500)
            if node[0] == 's':
                nx.draw_networkx_nodes(self.G, pos, nodelist=[node], node_color='orange', node_size=200)
        nx.draw_networkx_labels(self.G, pos)

        for edge in self.G.edges:
            if edge[0][0] == 'c' and edge[1][0] == 'c':
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], width=1.0, alpha=0.5, edge_color='red')
            elif edge[0][0] == 'c' or edge[1][0] == 'c':
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], width=1.0, alpha=0.5, edge_color='blue')
            else:
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], width=1.0, alpha=0.5, edge_color='green')

        labels = nx.get_edge_attributes(self.G, 'time')
        labels = {edge: f"{time:.2f}" for edge, time in labels.items()}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels, font_size=5)
        plt.title("Graph of time")
        plt.axis('off')

        # 成本图
        plt.figure()
        pos = nx.spring_layout(self.G)
        for node in self.G.nodes:
            if node[0] == 'c':
                nx.draw_networkx_nodes(self.G, pos, nodelist=[node], node_color='grey', node_size=500)
            if node[0] == 's':
                nx.draw_networkx_nodes(self.G, pos, nodelist=[node], node_color='orange', node_size=200)
        nx.draw_networkx_labels(self.G, pos)

        for edge in self.G.edges:
            if edge[0][0] == 'c' and edge[1][0] == 'c':
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], width=1.0, alpha=0.5, edge_color='red')
            elif edge[0][0] == 'c' or edge[1][0] == 'c':
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], width=1.0, alpha=0.5, edge_color='blue')
            else:
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], width=1.0, alpha=0.5, edge_color='green')

        labels = nx.get_edge_attributes(self.G, 'cost')
        labels = {edge: f"{cost:.2f}" for edge, cost in labels.items()}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels, font_size=5)
        plt.title("Graph of cost")
        plt.axis('off')
        plt.show()


    # 把locationID（str类型）转换成index（int类型）
    # 比如s1在locations里index为1，c0在locations里index为25，c2在locations里index为27......
    def id_index(self, ID_index):
        if type(ID_index) == str:
            if ID_index[0] == 's':
                index = int(ID_index[1:])
            else:
                index = int(ID_index[1:]) + parameters["station_num"]
            return index
        else:
            if ID_index < parameters["station_num"]:
                ID = 's' + str(ID_index)
            else:
                ID = 'c' + str(ID_index - parameters["station_num"])
            return ID

    def find_loc(self, id):
        for loc in self.locations:
            if loc.ID == id:
                return loc

    # 这个函数是准备用来处理express包裹的
    def pack_exp(self, src, dst):
        #chemin_min = self.chemin_time_matrix()
        src = int(src[1:])
        dst = int(dst[1:])
        route_min = self.route_min_time[src][dst]
        route_min_new = []
        for i in range(len(route_min)):
            route_min_new.append(self.id_index(route_min[i]))
        return route_min_new

    # 处理standard包裹
    def pack_normal(self, src, dst):
        #chemin_min = self.chemin_cost_matrix()
        src = int(src[1:])
        dst = int(dst[1:])
        route_min = self.route_min_cost[src][dst]
        route_min_new = []
        for i in range(len(route_min)):
            route_min_new.append(self.id_index(route_min[i]))
        return route_min_new


    # 处理包裹的模拟流程
    def pack_deal(self):
        pack_Delivered = parameters["packet_num"]

        while pack_Delivered != 0:
            pack_stat = self.events[0]
            pack = pack_stat[0]
            loc = self.find_loc(pack_stat[2])
            if pack_stat[-1] == "Created":
                pack_id = pack.ID
                if pack.category == 1:
                    pack_chemin = self.pack_exp(pack.src, pack.dst)[1:]
                else:
                    pack_chemin = self.pack_normal(pack.src, pack.dst)[1:]

                loc.pack_passby[pack_id] = pack.time_created
                self.tracking_info[pack_id] = [pack.src, pack.dst, pack.time_created,
                                               np.inf, [pack_stat[1], loc.ID, log_events[0]]]

                loc.update(pack.time_created)

                if loc.capa[0] > 0:
                    self.tracking_info[pack_id].append(
                        [pack.time_created, loc.ID, log_events[1]]
                    )
                    self.events.append([pack, pack.time_created + loc.delay, loc.ID, pack_chemin, "toSend"])
                    loc.capa[0]-=1
                else:
                    self.events.append([pack, pack.time_created+1, loc.ID, pack_chemin, "Waiting"])
                    loc.waiting.append(pack.ID)

            elif pack_stat[-1] == "Waiting":
                loc.update(pack_stat[1])
                if pack_stat[1]%1 != 0:
                    if loc.waiting:
                        buffer = PriorityQueue()
                        while pack.ID != loc.waiting[0]:
                            buffer.put(pack_stat)
                            pack_stat = self.events.get()
                            pack = pack_stat[0]
                        while not buffer.empty():
                            self.events.put(buffer.get())


                if loc.capa[0] > 0:
                    self.tracking_info[pack.ID].append([pack_stat[1], pack_stat[2], log_events[1]])
                    self.events.append([pack, pack_stat[1]+loc.delay, loc.ID, pack_stat[-2], "toSend"])
                    loc.capa[0] -= 1
                else:
                    self.events.append([pack, int(pack_stat[1])+1, loc.ID, pack_stat[-2], "Waiting"])

            elif pack_stat[-1] == "toSend":
                self.tracking_info[pack.ID].append([pack_stat[1], pack_stat[2], log_events[2]])
                if not pack_stat[-2]:
                    self.tracking_info[pack.ID][3] = pack_stat[1]
                    pack_Delivered -= 1
                    print(pack_Delivered)
                else:
                    dst = pack_stat[-2][0]
                    route_time = self.route_time_matrix[self.id_index(loc.ID)][self.id_index(dst)]
                    t_arrive = pack_stat[1] + route_time
                    self.events.append([pack, t_arrive, dst, pack_stat[-2], "Arrived"])

            else:
                self.tracking_info[pack.ID].append([pack_stat[1], loc.ID, log_events[0]])
                pack_stat[-2].pop(0)

                loc.update(pack_stat[1])
                if loc.capa[0] > 0:
                    self.tracking_info[pack.ID].append(
                        [pack_stat[1], loc.ID, log_events[1]])
                    self.events.append([pack, pack_stat[1] + loc.delay, loc.ID, pack_stat[-2], "toSend"])
                    loc.capa[0] -= 1
                else:
                    self.events.append([pack, pack_stat[1] + 1, loc.ID, pack_stat[-2], "Waiting"])
            self.events.pop(0)
            self.events.sort(key=lambda x: x[1])


    # 将tracking_info输出为csv
    def csv_tracking(self):
        column_names = ['ID', 'Src', 'Dst', 'TimeCreated', 'TimeDelivered', 'Log']
        df = pd.DataFrame(columns=column_names)
        list_tracking = list(self.tracking_info.keys())
        df['ID'] = list_tracking
        list_other = list(self.tracking_info.values())
        list_stat = []
        list_log = []
        for x in list_other:
            list_stat.append(x[:4])
            list_log.append(x[4:])
        df_stat = pd.DataFrame(list_stat, columns=['Column1', 'Column2', 'Column3', 'Column4'])
        #df_log = pd.DataFrame(list_log, columns=['Column1'])
        df[['Src', 'Dst', 'TimeCreated', 'TimeDelivered']] = df_stat
        df['Log'] = list_log
        #print(list_other)
        df.to_csv('Discrete_Time_Driven.csv', index=False)

class simulator2:
    def __init__(self, data):
        # 导入模拟的初始数据（input）
        self.locations = self.setlocation(data)
        self.routes = self.setroutes(data)
        self.packages = self.setpackages(data)
        self.G = self.route_graph(data) # 把route转换成graph，用去年图论的nx包
        self.tracking_info = {}
        self.on_route = []  # 在路上的包裹
        self.route_time_matrix = self.matrix("time")  # 路线的时间矩阵
        self.route_cost_matrix = self.matrix("cost")  # 路线的成本矩阵
        self.route_min_time = self.chemin_time_matrix()   # 最短时间路线矩阵
        self.route_min_cost = self.chemin_cost_matrix()   # 最小成本路线矩阵

    # 下面这几个set都是用来初始化用的
    def setlocation(self, data):
        locations = []
        for ID, pos, prop in zip(range(gen.parameters["station_num"]),
                                 data["station_pos"],
                                 data["station_prop"]):
            loc = Location("s" + str(ID), pos, prop[0], prop[1], prop[2])
            locations.append(loc)
        for ID, pos, prop in zip(range(gen.parameters["center_num"]),
                                 data["center_pos"],
                                 data["center_prop"]):
            loc = Location("c" + str(ID), pos, prop[0], prop[1], prop[2])
            locations.append(loc)
        return locations

    def setroutes(self, data):
        routes = []
        for src, dst, time, cost in data["edges"]:
            route = Route(src, dst, time, cost)
            routes.append(route)
        return routes

    def setpackages(self, data):
        packages = []
        for pkg_data in data["packets"]:
            pkg_id = pkg_data[0]
            pkg = Package(pkg_data[0], pkg_data[1], pkg_data[2], pkg_data[3], pkg_data[4])
            packages.append(pkg)
        return packages

    # 把route数据转换成图
    def route_graph(self, routes):
        G = nx.DiGraph()
        for route in self.routes:
            G.add_edge(route.src, route.dst, time=route.time, cost=route.cost)
        return G

    # 返回一个邻接矩阵（表示两个location之间的time 或者 cost）
    def matrix(self, time_cost):
        matrix = np.full((parameters["station_num"]+parameters["center_num"], parameters["station_num"]+parameters["center_num"]), np.inf)
        np.fill_diagonal(matrix, 0)
        # time矩阵
        if time_cost == 'time':
            for route in self.routes:
                if route.src[0] == 's':
                    if route.dst[0] == 's':
                        matrix[int(route.src[1:])][int(route.dst[1:])] = route.time
                    else:
                        matrix[int(route.src[1:])][int(route.dst[1:])+parameters["station_num"]] = route.time
                else:
                    if route.dst[0] == 's':
                        matrix[int(route.src[1:])+parameters["station_num"]][int(route.dst[1:])] = route.time
                    else:
                        matrix[int(route.src[1:]) + parameters["station_num"]][int(route.dst[1:])+parameters["station_num"]] = route.time
        # cost矩阵
        else:
            for route in self.routes:
                if route.src[0] == 's':
                    if route.dst[0] == 's':
                        matrix[int(route.src[1:])][int(route.dst[1:])] = route.cost
                    else:
                        matrix[int(route.src[1:])][int(route.dst[1:])+parameters["station_num"]] = route.cost
                else:
                    if route.dst[0] == 's':
                        matrix[int(route.src[1:])+parameters["station_num"]][int(route.dst[1:])] = route.cost
                    else:
                        matrix[int(route.src[1:])+parameters["station_num"]][int(route.dst[1:])+parameters["station_num"]] = route.cost
        return matrix

    # 返回一个时间最短路径矩阵，比如第二行第五列代表s2到s5的最短路径
    def chemin_time_matrix(self):
        time_matrix = self.route_time_matrix
        loc_delay = [loc.delay for loc in self.locations]
        chemin_time = []
        for src in self.locations:
            src = int(src.ID[1:])
            chemin = self.dijkstra(time_matrix, loc_delay, src)
            chemin_time.append(chemin)
        return chemin_time

    # 同理，cost最少的路径
    def chemin_cost_matrix(self):
        cost_matrix = self.route_cost_matrix
        loc_cost = [loc.cost for loc in self.locations]
        chemin_cost = []
        for src in self.locations:
            src = int(src.ID[1:])
            chemin = self.dijkstra(cost_matrix, loc_cost, src)
            chemin_cost.append(chemin)
        return chemin_cost
    # ------------------初始化函数到这结束------------------#

    # 用nx包画站和路径的图
    def draw_route(self):

        # 时间图
        plt.figure()
        pos = nx.spring_layout(self.G)
        for node in self.G.nodes:
            if node[0] == 'c':
                nx.draw_networkx_nodes(self.G, pos, nodelist=[node], node_color='grey', node_size=500)
            if node[0] == 's':
                nx.draw_networkx_nodes(self.G, pos, nodelist=[node], node_color='orange', node_size=200)
        nx.draw_networkx_labels(self.G, pos)

        for edge in self.G.edges:
            if edge[0][0] == 'c' and edge[1][0] == 'c':
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], width=1.0, alpha=0.5, edge_color='red')
            elif edge[0][0] == 'c' or edge[1][0] == 'c':
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], width=1.0, alpha=0.5, edge_color='blue')
            else:
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], width=1.0, alpha=0.5, edge_color='green')

        labels = nx.get_edge_attributes(self.G, 'time')
        labels = {edge: f"{time:.2f}" for edge, time in labels.items()}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels, font_size=5)
        plt.title("Graph of time")
        plt.axis('off')

        # 成本图
        plt.figure()
        pos = nx.spring_layout(self.G)
        for node in self.G.nodes:
            if node[0] == 'c':
                nx.draw_networkx_nodes(self.G, pos, nodelist=[node], node_color='grey', node_size=500)
            if node[0] == 's':
                nx.draw_networkx_nodes(self.G, pos, nodelist=[node], node_color='orange', node_size=200)
        nx.draw_networkx_labels(self.G, pos)

        for edge in self.G.edges:
            if edge[0][0] == 'c' and edge[1][0] == 'c':
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], width=1.0, alpha=0.5, edge_color='red')
            elif edge[0][0] == 'c' or edge[1][0] == 'c':
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], width=1.0, alpha=0.5, edge_color='blue')
            else:
                nx.draw_networkx_edges(self.G, pos, edgelist=[edge], width=1.0, alpha=0.5, edge_color='green')

        labels = nx.get_edge_attributes(self.G, 'cost')
        labels = {edge: f"{cost:.2f}" for edge, cost in labels.items()}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels, font_size=5)
        plt.title("Graph of cost")
        plt.axis('off')
        plt.show()

    # 最短路径dijkstra算法，只返回路径不返回最短时间or成本 （看seb的课
    def dijkstra(self, M, loc_M, s0):
        sk = s0
        n = len(M)
        delta = [np.inf] * n
        delta[s0] = 0
        mark = [False] * n
        mark[s0] = True
        chemin = [[]] * n
        chemin[s0] = [s0]
        while sum(mark) < n:
            for t in range(n):
                if M[sk][t] != 0 and delta[sk] + M[sk][t] + loc_M[t] < delta[t]:
                    delta[t] = delta[sk] + M[sk][t] + loc_M[t]
                    chemin[t] = chemin[sk] + [t]
            delta_min = np.inf
            for s in range(n):
                if not (mark[s]) and delta[s] < delta_min:
                    sk = s
                    delta_min = delta[sk]
            mark[sk] = True
        return chemin

    # 把locationID（str类型）转换成index（int类型）
    # 比如s1在locations里index为1，c0在locations里index为25，c2在locations里index为27......
    def s_or_c(self, ID):
        if ID[0] == 's':
            index = int(ID[1:])
        else:
            index = int(ID[1:]) + parameters["station_num"]
        return index

    # 把index转换成ID，int类型到str
    def int_ID(self, index):
        if index < parameters["station_num"]:
            ID = 's' + str(index)
        else:
            ID = 'c' + str(index - parameters["station_num"])
        return ID

    # 这个函数是准备用来处理express包裹的
    def pack_exp(self, src, dst):
        #chemin_min = self.chemin_time_matrix()
        src = int(src[1:])
        dst = int(dst[1:])
        route_min = self.route_min_time[src][dst]
        route_min_new = []
        for i in range(len(route_min)):
            route_min_new.append(self.int_ID(route_min[i]))
        return route_min_new

    # 处理standard包裹
    def pack_normal(self, src, dst):
        #chemin_min = self.chemin_cost_matrix()
        src = int(src[1:])
        dst = int(dst[1:])
        route_min = self.route_min_cost[src][dst]
        route_min_new = []
        for i in range(len(route_min)):
            route_min_new.append(self.int_ID(route_min[i]))
        return route_min_new

    # 处理包裹的模拟流程
    def pack_deal(self):
        t = 0
        on_route = []
        packages_copy = self.packages
        pack_Delivered = parameters["packet_num"]
        while pack_Delivered != 0:
            t += 1
            # 处理新生成包裹
            # buffer[arrived]里元素是 [pack(class), 到站时间，剩余路径]
            while (packages_copy):
                if packages_copy[0].time_created <= t:
                    if packages_copy[0].category == 1:
                        pack_chemin = self.pack_exp(packages_copy[0].src, packages_copy[0].dst)[1:]
                    else:
                        pack_chemin = self.pack_normal(packages_copy[0].src, packages_copy[0].dst)[1:]
                    self.locations[int(packages_copy[0].src[1:])].buffer['arrived'].append(
                        [packages_copy[0], packages_copy[0].time_created, pack_chemin])
                    self.locations[int(packages_copy[0].src[1:])].pack_passby[packages_copy[0].ID] = packages_copy[0].time_created
                    self.tracking_info[packages_copy[0].ID] = [packages_copy[0].src, packages_copy[0].dst,
                                                               packages_copy[0].time_created, np.inf,
                                                               [packages_copy[0].time_created,
                                                                self.locations[int(packages_copy[0].src[1:])].ID,
                                                                log_events[0]]]
                    packages_copy.pop(0)
                else:
                    break

            # 处理到站包裹
            # on_route里元素是 [pack（类），下一站，到达时间，剩余路径]
            on_route_copy = [x for x in on_route]
            for pack_arrived in on_route:
                if pack_arrived[2] < t:
                    pack_arrived[3].pop(0)
                    if pack_arrived[1][0] == 'c':
                        self.locations[int(pack_arrived[1][1:])+parameters["station_num"]].buffer['arrived'].append(
                            [pack_arrived[0], pack_arrived[2], pack_arrived[3]])
                        self.locations[int(pack_arrived[1][1:])+parameters["station_num"]].pack_passby[
                            pack_arrived[0].ID] = pack_arrived[2]
                        self.tracking_info[pack_arrived[0].ID].append(
                            [pack_arrived[2], pack_arrived[1], log_events[0]])
                    else:
                        self.locations[int(pack_arrived[1][1:])].buffer['arrived'].append(
                            [pack_arrived[0], pack_arrived[2], pack_arrived[3]])
                        self.locations[int(pack_arrived[1][1:])].pack_passby[pack_arrived[0].ID] = \
                            pack_arrived[2]
                        self.tracking_info[pack_arrived[0].ID].append(
                            [pack_arrived[2], pack_arrived[1], log_events[0]])
                    on_route_copy.remove(pack_arrived)
                else:
                    continue
            on_route = [x for x in on_route_copy]

            # 处理站内包裹，先把上个时间的send出去
            # 找哪些要send
            for loc in self.locations:
                # pack_pro: [pack(class), 开始处理时间，剩余路径]
                # 先排个序
                loc.buffer['processing'].sort(key=lambda x: x[1], reverse=True)
                loc.buffer['arrived'].sort(key=lambda x: x[1])
                loc.buffer['ready_to_send'].sort(key=lambda  x:x[1], reverse=True)
                for i in range(len(loc.buffer['processing'])-1, -1, -1):
                    pack_pro = loc.buffer['processing'][i]
                    if t - pack_pro[1] > loc.delay:
                        time_send = pack_pro[1]+loc.delay
                        loc.buffer['ready_to_send'].append([pack_pro[0], time_send, pack_pro[2]])
                        loc.buffer['processing'].pop()
                    else:
                        break

                # 发射！
                # pack_tosend 为 [pack(class), 发出时间，剩余路径]
                for i in range(len(loc.buffer['ready_to_send'])-1, -1, -1):
                    pack_tosend = loc.buffer['ready_to_send'][i]
                    self.tracking_info[pack_tosend[0].ID].append([pack_tosend[1], loc.ID, log_events[2]])
                    # 如果这就是终点
                    if not pack_tosend[2]:
                        self.tracking_info[pack_tosend[0].ID][3] = pack_tosend[1]
                        self.tracking_info[pack_tosend[0].ID][3] = pack_tosend[1]
                        pack_Delivered -= 1
                    # 如果还没走完
                    else:
                        next_loc = pack_tosend[2][0]
                        route_time = self.route_time_matrix[self.s_or_c(loc.ID)][self.s_or_c(next_loc)]
                        time_arrive = pack_tosend[1] + route_time
                        on_route.append([pack_tosend[0], next_loc, time_arrive, pack_tosend[2]])
                    loc.buffer['ready_to_send'].pop()

                # 把到达的包裹进行处理
                # buffer[arrived]里元素是 [pack(class), 到站时间，剩余路径]
                # pack_pro: [pack(class), 开始处理时间，剩余路径]
                if len(loc.buffer['arrived']) <= loc.throughput:
                    # 如果目前到站的没超过站的最大处理量
                    for pack_arr in loc.buffer['arrived']:
                        waiting_time = (t - pack_arr[1]) // 1
                        pack_arr[1] = pack_arr[1]+waiting_time
                        loc.buffer['processing'].append(pack_arr)
                        self.tracking_info[pack_arr[0].ID].append(
                            [pack_arr[1]+waiting_time, loc.ID, log_events[1]])
                    loc.buffer['arrived'].clear()
                else:
                    # 如果超过了
                    buffer_arrived_copy = [x for x in loc.buffer['arrived']]
                    for pack_notdeal in loc.buffer['arrived'][loc.throughput:]:
                        pack_notdeal[1] += 1
                    for pack_arr in loc.buffer['arrived'][:loc.throughput]:
                        waiting_time = (t - pack_arr[1]) // 1
                        pack_arr[1] = pack_arr[1] + waiting_time
                        loc.buffer['processing'].append(pack_arr)
                        self.tracking_info[pack_arr[0].ID].append(
                            [pack_arr[1] + waiting_time, loc.ID, log_events[1]])
                        buffer_arrived_copy.remove(pack_arr)
                    loc.buffer['arrived'] = [x for x in buffer_arrived_copy]

    # 将tracking_info输出为csv
    def csv_tracking(self):
        column_names = ['ID', 'Src', 'Dst', 'TimeCreated', 'TimeDelivered', 'Log']
        df = pd.DataFrame(columns=column_names)
        list_tracking = list(self.tracking_info.keys())
        df['ID'] = list_tracking
        list_other = list(self.tracking_info.values())
        list_stat = []
        list_log = []
        for x in list_other:
            list_stat.append(x[:4])
            list_log.append(x[4:])
        df_stat = pd.DataFrame(list_stat, columns=['Column1', 'Column2', 'Column3', 'Column4'])
        #df_log = pd.DataFrame(list_log, columns=['Column1'])
        df[['Src', 'Dst', 'TimeCreated', 'TimeDelivered']] = df_stat
        df['Log'] = list_log
        #print(list_other)
        df.to_csv('Event_Driven.csv', index=False)


if __name__ == '__main__':
    # 生成模拟数据
    data = gen.data_gen()

    # 导入数据
    sim = simulator(data)
    sim2 = simulator2(data)

    sim.pack_deal()
    sim2.pack_deal()

    sim.csv_tracking()
    sim2.csv_tracking()