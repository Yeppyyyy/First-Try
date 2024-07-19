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
    "packet_num": 5000,
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
        self.buffer_num = 0
        self.pack_passby = PriorityQueue()
        self.capa = [throughput, 0]
        self.t_waiting = 0
        self.pack_arrive = []

    def update_t(self):
        if self.buffer_num > 0:
            self.t_waiting = self.buffer_num // self.throughput

    # 更新timetick和处理量
    def update(self, t):
        if self.capa[1] < int(t):
            self.capa[1] = int(t)
            self.capa[0] = self.throughput

    def waiting_At_t(self, t):
        num_left = self.buffer_num
        for i in self.pack_arrive:
            if i <= t:
                num_left += 1
            else:
                break
        num_left = max(0, num_left-t*self.throughput)
        waiting_t = num_left // self.throughput
        return waiting_t



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
        self.arrived_time = 0

    def __lt__(self, other):
        return self.arrived_time < other.arrived_time

# Location
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
        self.tosend_t = 0
        self.packageID = [str(x.ID) for x in self.packages]
        self.fig, self.ax = self.bg(data)

    def bg(self, data):
        return data["fig"], data["ax"]

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

    def chemin_time_matrix(self):
        time_matrix = self.route_time_matrix
        loc_delay = [loc.delay for loc in self.locations]
        chemin_time = []
        for src in self.locations:
            src = int(src.ID[1:])
            chemin = self.dijkstra(time_matrix, loc_delay, src)
            chemin_time.append(chemin)
        return chemin_time


    # 初始化事件池
    def init_events(self):
        events = PriorityQueue()
        for pack in self.packages:
            pack.arrived_time = pack.time_created
            events.put([pack.time_created, pack, pack.src, "Created"])
            loc = self.find_loc(pack.src)
            loc.pack_arrive.append(pack.time_created)
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


    def update_M(self, tt):
        M_new = self.route_time_matrix.copy()
        for i1 in range(len(self.route_time_matrix)):
            x = self.route_time_matrix[i1]
            for i2 in range(len(x)):
                if x[i2] != np.inf:
                    time_route = x[i2]
                    time_waiting = self.locations[i2].waiting_At_t(tt+time_route)
                    if time_waiting > time_route:
                        M_new[i1][i2] = time_waiting
        return M_new


    # 这个函数是准备用来处理express包裹的
    def pack_exp(self, src, dst):
        loc_Delay = [x.delay for x in self.locations]
        tt = 0
        M = self.update_M(tt)
        sk = self.id_index(src)
        n = len(M)
        delta = [np.inf] * n
        delta[self.id_index(src)] = 0
        mark = [False] * n
        mark[self.id_index(src)] = True
        chemin = [[]] * n
        chemin[self.id_index(src)] = [src]
        while sum(mark) < n:
            for t in range(n):
                if M[sk][t] != 0 and delta[sk] + M[sk][t] + loc_Delay[t] < delta[t]:
                    delta[t] = delta[sk] + M[sk][t] + loc_Delay[t]
                    chemin[t] = chemin[sk] + [t]
            delta_min = np.inf
            for s in range(n):
                if not (mark[s]) and delta[s] < delta_min:
                    sk = s
                    delta_min = delta[sk]

            mark[sk] = True
            tt = delta_min
            M = self.update_M(tt)
        dst = self.id_index(dst)
        chemin_min = []
        for x in chemin[dst]:
            chemin_min.append(self.id_index(x))
        return chemin_min


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
            pack_stat = self.events.get()
            pack = pack_stat[1]
            loc = self.find_loc(pack_stat[2])
            if pack.category == 0:
                if pack_stat[-1] == "Created":
                    pack_id = pack.ID
                    pack_chemin = self.pack_normal(pack.src, pack.dst)[1:]

                    loc.pack_passby.put([pack_stat[0], pack.ID])
                    self.tracking_info[pack_id] = [pack.src, pack.dst, pack.time_created,
                                                   np.inf, [pack_stat[0], loc.ID, log_events[0]]]

                    loc.update(pack.time_created)

                    if loc.capa[0] > 0:
                        self.tracking_info[pack_id].append(
                            [pack.time_created, loc.ID, log_events[1]]
                        )
                        self.events.put([pack.time_created + loc.delay, pack, loc.ID, pack_chemin, "toSend"])
                        loc.capa[0]-=1
                    else:
                        self.events.put([int(pack.time_created)+1, pack, loc.ID, pack_chemin, "Waiting"])
                        loc.buffer_num += 1

                elif pack_stat[-1] == "Waiting":
                    loc.update(pack_stat[0])
                    if loc.capa[0] > 0:
                        self.tracking_info[pack.ID].append([pack_stat[0], pack_stat[2], log_events[1]])
                        self.events.put([pack_stat[0]+loc.delay, pack, loc.ID, pack_stat[-2], "toSend"])
                        loc.capa[0] -= 1
                        loc.buffer_num -= 1
                    else:
                        self.events.put([pack_stat[0] + 1, pack, loc.ID, pack_stat[-2], "Waiting"])


                elif pack_stat[-1] == "toSend":
                    self.tracking_info[pack.ID].append([pack_stat[0], pack_stat[2], log_events[2]])
                    if not pack_stat[-2]:
                        self.tracking_info[pack.ID][3] = pack_stat[0]
                        pack_Delivered -= 1
                        #print(pack_Delivered)
                    else:
                        dst = pack_stat[-2][0]
                        route_time = self.route_time_matrix[self.id_index(loc.ID)][self.id_index(dst)]
                        t_arrive = pack_stat[0] + route_time
                        pack.arrived_time = t_arrive
                        self.events.put([t_arrive, pack, dst, pack_stat[-2], "Arrived"])
                        dst = self.find_loc(dst)
                        dst.pack_arrive.append(t_arrive)

                else:
                    self.tracking_info[pack.ID].append([pack_stat[0], loc.ID, log_events[0]])
                    pack_stat[-2].pop(0)
                    loc.pack_passby.put([pack_stat[0], pack.ID])
                    loc.update(pack_stat[0])
                    loc.pack_arrive.pop(0)
                    if loc.capa[0] > 0:
                        self.tracking_info[pack.ID].append(
                            [pack_stat[0], loc.ID, log_events[1]])
                        self.events.put([pack_stat[0] + loc.delay, pack, loc.ID, pack_stat[-2], "toSend"])
                        loc.capa[0] -= 1
                    else:
                        loc.buffer_num += 1
                        self.events.put([int(pack_stat[0]) + 1, pack, loc.ID, pack_stat[-2], "Waiting"])
            else:
                if pack_stat[-1] == "Created":
                    pack_id = pack.ID
                    pack_chemin = self.pack_exp(pack.src, pack.dst)[1:]

                    loc.pack_passby.put([pack_stat[0], pack.ID])
                    self.tracking_info[pack_id] = [pack.src, pack.dst, pack.time_created,
                                                   np.inf, [pack_stat[0], loc.ID, log_events[0]]]

                    loc.update(pack.time_created)

                    if loc.capa[0] > 0:
                        self.tracking_info[pack_id].append(
                            [pack.time_created, loc.ID, log_events[1]]
                        )
                        self.events.put([pack.time_created + loc.delay, pack, loc.ID, pack_chemin[0], "toSend"])
                        loc.capa[0]-=1
                    else:
                        self.events.put([int(pack.time_created)+1, pack, loc.ID, pack_chemin[0], "Waiting"])
                        loc.buffer_num += 1

                elif pack_stat[-1] == "Waiting":
                    loc.update(pack_stat[0])
                    if loc.capa[0] > 0:
                        self.tracking_info[pack.ID].append([pack_stat[0], pack_stat[2], log_events[1]])
                        self.events.put([pack_stat[0]+loc.delay, pack, loc.ID, pack_stat[-2][0], "toSend"])
                        loc.capa[0] -= 1
                        loc.buffer_num -= 1
                    else:
                        self.events.put([pack_stat[0] + 1, pack, loc.ID, pack_stat[-2][0], "Waiting"])


                elif pack_stat[-1] == "toSend":
                    self.tosend_t = pack_stat[0] - self.tosend_t
                    for location in self.locations:
                        location.pack_arrive= [i - self.tosend_t for i in location.pack_arrive]
                    self.tracking_info[pack.ID].append([pack_stat[0], pack_stat[2], log_events[2]])
                    if pack.dst == loc.ID:
                        self.tracking_info[pack.ID][3] = pack_stat[0]
                        pack_Delivered -= 1
                        #print(pack_Delivered)
                    else:

                        pack_chemin = self.pack_exp(pack_stat[2], pack.dst)[1:]
                        dst = pack_chemin[0]
                        route_time = self.route_time_matrix[self.id_index(loc.ID)][self.id_index(dst)]

                        t_arrive = pack_stat[0] + route_time
                        pack.arrived_time = t_arrive
                        self.events.put([t_arrive, pack, dst, pack_stat[-2], "Arrived"])
                        dst = self.find_loc(dst)
                        dst.pack_arrive.append(t_arrive)


                else:
                    self.tracking_info[pack.ID].append([pack_stat[0], loc.ID, log_events[0]])
                    loc.update(pack_stat[0])
                    loc.pack_arrive.pop(0)
                    loc.pack_passby.put([pack_stat[0], pack.ID])
                    if loc.capa[0] > 0:
                        self.tracking_info[pack.ID].append(
                            [pack_stat[0], loc.ID, log_events[1]])
                        self.events.put([pack_stat[0] + loc.delay, pack, loc.ID, pack_stat[-2], "toSend"])
                        loc.capa[0] -= 1
                    else:
                        loc.buffer_num += 1
                        self.events.put([int(pack_stat[0]) + 1, pack, loc.ID, pack_stat[-2], "Waiting"])

            #print(len(self.events))



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
        df.to_csv('Optimize.csv', index=False)
        print("The tracking information is saved in optimize.csv.")





class simulator1:
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
        self.packageID = [str(x.ID) for x in self.packages]
        self.fig, self.ax = self.bg(data)

    def bg(self, data):
        return data["fig"], data["ax"]

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
        events = PriorityQueue()
        for pack in self.packages:
            pack.arrived_time = pack.time_created
            events.put([pack.time_created, pack, pack.src, "Created"])
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
            pack_stat = self.events.get()
            pack = pack_stat[1]
            loc = self.find_loc(pack_stat[2])
            if pack_stat[-1] == "Created":
                pack_id = pack.ID
                if pack.category == 1:
                    pack_chemin = self.pack_exp(pack.src, pack.dst)[1:]
                else:
                    pack_chemin = self.pack_normal(pack.src, pack.dst)[1:]

                self.tracking_info[pack_id] = [pack.src, pack.dst, pack.time_created,
                                               np.inf, [pack_stat[0], loc.ID, log_events[0]]]
                loc.pack_passby.put([pack_stat[0], pack.ID])

                loc.update(pack.time_created)

                if loc.capa[0] > 0:
                    self.tracking_info[pack_id].append(
                        [pack.time_created, loc.ID, log_events[1]]
                    )
                    self.events.put([pack.time_created + loc.delay, pack, loc.ID, pack_chemin, "toSend"])
                    loc.capa[0]-=1
                else:
                    self.events.put([pack.time_created+1, pack, loc.ID, pack_chemin, "Waiting"])

            elif pack_stat[-1] == "Waiting":
                loc.update(pack_stat[0])

                if loc.capa[0] > 0:
                    self.tracking_info[pack.ID].append([pack_stat[0], pack_stat[2], log_events[1]])
                    self.events.put([pack_stat[0]+loc.delay, pack, loc.ID, pack_stat[-2], "toSend"])
                    loc.capa[0] -= 1
                else:
                    self.events.put([pack_stat[0]+1, pack, loc.ID, pack_stat[-2], "Waiting"])

            elif pack_stat[-1] == "toSend":
                self.tracking_info[pack.ID].append([pack_stat[0], pack_stat[2], log_events[2]])
                if not pack_stat[-2]:
                    self.tracking_info[pack.ID][3] = pack_stat[0]
                    pack_Delivered -= 1
                    #print(pack_Delivered)
                else:
                    dst = pack_stat[-2][0]
                    route_time = self.route_time_matrix[self.id_index(loc.ID)][self.id_index(dst)]
                    t_arrive = pack_stat[0] + route_time
                    #pack.arrived_time = t_arrive
                    self.events.put([t_arrive, pack, dst, pack_stat[-2], "Arrived"])

            else:
                self.tracking_info[pack.ID].append([pack_stat[0], loc.ID, log_events[0]])
                pack_stat[-2].pop(0)
                loc.pack_passby.put([pack_stat[0], pack.ID])

                loc.update(pack_stat[0])
                if loc.capa[0] > 0:
                    self.tracking_info[pack.ID].append(
                        [pack_stat[0], loc.ID, log_events[1]])
                    self.events.put([pack_stat[0] + loc.delay, pack, loc.ID, pack_stat[-2], "toSend"])
                    loc.capa[0] -= 1
                else:
                    self.events.put([pack_stat[0] + 1, pack, loc.ID, pack_stat[-2], "Waiting"])


            #print(len(self.events))


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
        df.to_csv('No_optimize.csv', index=False)
        print("The tracking information is saved in tracking_info.csv.")



if __name__ == '__main__':
    # 生成模拟数据
    data = gen.data_gen()

    # 导入数据
    sim = simulator(data)
    sim1 = simulator1(data)

    sim.pack_deal()
    sim1.pack_deal()

    num_fast = 0
    num_slow = 0
    a = 0
    b = 0
    c = 0
    d = 0
    for pack in sim.packages:
            if sim.tracking_info[pack.ID][3] < sim1.tracking_info[pack.ID][3]:
                a += 1
                num_fast+= sim.tracking_info[pack.ID][3] - sim1.tracking_info[pack.ID][3]
            elif sim.tracking_info[pack.ID][3] > sim1.tracking_info[pack.ID][3]:
                b += 1
                num_slow+=sim.tracking_info[pack.ID][3] - sim1.tracking_info[pack.ID][3]
            else:
                pass



    print(f"The total number of slower packets is {a}.")
    print(f"The total number of faster packets is {b}.")
    print(f"Saved time: {abs(num_fast+num_slow)}")
    print(f"The average saved time of faster packet is {abs(num_fast)/a}.")