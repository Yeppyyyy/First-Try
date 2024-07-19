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
        self.process =[]
        self.soon_arrive = []
        #self.virtualprocess = []

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
        self.cheminn = []
    def __lt__(self, other):
        return self.ID < other.ID

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
            if pack.category == 1:
                pack_chemin = self.pack_exp(pack.src, pack.dst)
            else:
                pack_chemin = self.pack_normal(pack.src, pack.dst)
            events.put([pack.time_created, pack, pack.src, pack_chemin, "ARRIVED"])##重要 src是出发地
            #print(pack.src,pack.dst,pack_chemin)
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
        for pack in self.packages:
            self.tracking_info[pack.ID] = [pack.src, pack.dst, pack.time_created]
        # 假设loc.delay是处理一个包裹的时间
        delivering= parameters["packet_num"]
        while not self.events.empty() :  # 优先队列不为空

            pack_stat = self.events.get()
            #print(pack_stat)# 弹出第一个事件
            time = pack_stat[0]  # 时间
            pack = pack_stat[1]  # pack内容
            chemin = pack_stat[3]
            #print(chemin)
            #print((pack.src,pack.dst))
            loc = self.find_loc(pack_stat[3][0])  # 回到大类loc
            if pack_stat[-1] == "ARRIVED":
                #print(0)
                self.tracking_info[pack.ID].append([time, loc.ID, "ARRIVED"])
                if len(loc.process) >= loc.capa[0]:  # 如果这地方在处理的和待处理的长度大于capa了
                    if pack.category == 0:  # 普通包裹 只要计算等待时间就可以了
                        timee = loc.delay + loc.process[-loc.capa[0]][1]  # 前capacity个处理完的时间
                        loc.process.append([pack, timee])
                        self.events.put([timee, pack, loc.ID, chemin, "PROCESSING"])
                    else:
                        #print(6666)
                        timee = loc.delay + loc.process[-loc.capa[0]][1]
                        loc.process.append([pack, timee])  # 假设express包裹插在最后
                        i = len(loc.process) - 2
                        kk = []  # 存event里需要变换的ID
                        mm = []  # 存event里需要变换的ID的新时间
                        timefinal = timee
                        rate = int((len(loc.process) - loc.capa[0])*0.5)
                        fois = 0
                        while (loc.process[i][0]).category == 0 and i >= loc.capa[0] and fois <= rate:
                            #print(0)
                            # 逆向遍历，遇到处理中或者express的包裹则停止
                            timeexchange = loc.process[i + 1][1]
                            loc.process[i + 1][1] = loc.process[i][1]
                            mm.append(timeexchange)  # mm记录standard包裹的新时间
                            timefinal = loc.process[i][1]  # 最后这个express包裹落到的位置
                            loc.process[i][1] = timeexchange  # 对于这个standard和express交换时间

                            temp = loc.process[i + 1]
                            loc.process[i + 1] = loc.process[i]
                            kk.append((loc.process[i][0]).ID)  # mm记录standard包裹的编号
                            loc.process[i] = temp  # 交换pack和timee整体，冒泡过程
                            i = i - 1
                            fois = fois + 1
                        # mm kk是一一对应的
                        if i != len(loc.process) - 2:
                            t = 0
                            pool = []
                            #print(timee)
                            while t <= timee and len(pool) < delivering-1:  # 改动优先队列，只能一个个弹出来，判断要不要改，再加回去只有这些要改 这里可能有死循环
                                temp = self.events.get()
                                #print(6667)
                                t = temp[0]
                                #print(t)# 时间
                                packk = temp[1]  # 包裹
                                cheminnew = temp[3]
                                if (temp[1].ID) in kk:  # 判断是否需要改动 当我们确保队列里只存在包裹的一个状态，才能这么做
                                    #print(11)
                                    index = kk.index(temp[1].ID)
                                    pool.append([mm[index], packk, loc.ID, cheminnew, "PROCESSING"])  # 因为只有这个location的会被改动 chemin
                                    #print(len(pool))
                                    #print(timee)
                                    #print(delivering)
                                else:
                                    pool.append(temp)
                                    #print(len(pool))
                                    #print(timee)
                                    #print(delivering)# 不需要改动则放回去 可能永远弹出一个包裹
                                    # 看起来这部分非常耗时，实际上没几次需要改
                            #print(6668)
                            #到不了这个位置
                            for ele in pool:
                                self.events.put(ele)
                        self.events.put([timefinal, pack, loc.ID, chemin, "PROCESSING"])  # 加进这个express包裹

                else:  # len(loc.process) < loc.capa 如果这地方在处理的和待处理的长小于capa
                    loc.process.append([pack, time])  # 直接加进去就可以了,因为可以即刻处理
                    self.events.put([time, pack, loc.ID, chemin, "PROCESSING"])  # event也可以直接放进去


            elif pack_stat[-1] == "PROCESSING":
                #print(1)
                self.tracking_info[pack.ID].append([time, loc.ID, "PROCESSING"])
                self.events.put([time + loc.delay, pack, loc.ID, chemin, "SEND"])  # 加一份处理时间放进去就可以了

            else:  # send的情况
                #print(2)
                self.tracking_info[pack.ID].append([time, loc.ID, "SEND"])
                if pack.dst == pack_stat[2]:# 到地方了，就不加回去了 locprocess
                    delivering = delivering -1
                    for k in loc.process:
                        if (k[0]).ID == pack.ID:
                            (loc.process).remove(k)  # 维护loc.process 发送就代表不再是processing状态了
                            break
                else:
                    for k in loc.process:
                        if (k[0]).ID == pack.ID:
                            (loc.process).remove(k)  # 维护loc.process 发送就代表不再是processing状态了
                            break  # 能不能remove0
                    #print(pack.src,pack.dst)
                    #print(self.tracking_info[pack.ID])
                    #print(loc.ID)
                    #print(chemin)
                    #print(pack.dst)
                    #print(chemin[0], chemin[1])
                    #print(self.id_index(chemin[0]),self.id_index(chemin[1]))
                    route_time = self.route_time_matrix[self.id_index(chemin[0])][self.id_index(chemin[1])]
                    t_arrive = time + route_time
                    #print(chemin[1:])
                    #print(pack.dst)
                    self.events.put([t_arrive, pack, loc.ID, chemin[1:], "ARRIVED"])
        #print(1)

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
        df.to_csv('Tracking_info.csv', index=False)

# event-driven
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
            if pack.category == 1:
                pack_chemin = self.pack_exp(pack.src, pack.dst)
            else:
                pack_chemin = self.pack_normal(pack.src, pack.dst)
            events.put([pack.time_created, pack, pack.src, pack_chemin, "ARRIVED"])##重要 src是出发地
            #print(pack.src,pack.dst,pack_chemin)
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
        for pack in self.packages:
            self.tracking_info[pack.ID] = [pack.src, pack.dst, pack.time_created]
        # 假设loc.delay是处理一个包裹的时间
        delivering= parameters["packet_num"]
        while not self.events.empty() :  # 优先队列不为空

            pack_stat = self.events.get()
            #print(pack_stat)# 弹出第一个事件
            time = pack_stat[0]  # 时间
            pack = pack_stat[1]  # pack内容
            chemin = pack_stat[3]
            #print(chemin)
            #print((pack.src,pack.dst))
            loc = self.find_loc(pack_stat[3][0])  # 回到大类loc
            if pack_stat[-1] == "ARRIVED":
                #print(0)
                self.tracking_info[pack.ID].append([time, loc.ID, "ARRIVED"])
                if len(loc.process) >= loc.capa[0]:  # 如果这地方在处理的和待处理的长度大于capa了  # 普通包裹 只要计算等待时间就可以了
                        timee = loc.delay + loc.process[-loc.capa[0]][1]  # 前capacity个处理完的时间
                        loc.process.append([pack, timee])
                        self.events.put([timee, pack, loc.ID, chemin, "PROCESSING"])
                else:  # len(loc.process) < loc.capa 如果这地方在处理的和待处理的长小于capa
                    loc.process.append([pack, time])  # 直接加进去就可以了,因为可以即刻处理
                    self.events.put([time, pack, loc.ID, chemin, "PROCESSING"])  # event也可以直接放进去


            elif pack_stat[-1] == "PROCESSING":
                #print(1)
                self.tracking_info[pack.ID].append([time, loc.ID, "PROCESSING"])
                self.events.put([time + loc.delay, pack, loc.ID, chemin, "SEND"])  # 加一份处理时间放进去就可以了

            else:  # send的情况
                #print(2)
                self.tracking_info[pack.ID].append([time, loc.ID, "SEND"])
                if pack.dst == pack_stat[2]:# 到地方了，就不加回去了 locprocess
                    delivering = delivering -1
                    for k in loc.process:
                        if (k[0]).ID == pack.ID:
                            (loc.process).remove(k)  # 维护loc.process 发送就代表不再是processing状态了
                            break
                else:
                    for k in loc.process:
                        if (k[0]).ID == pack.ID:
                            (loc.process).remove(k)  # 维护loc.process 发送就代表不再是processing状态了
                            break  # 能不能remove0
                    #print(pack.src,pack.dst)
                    #print(self.tracking_info[pack.ID])
                    #print(loc.ID)
                    #print(chemin)
                    #print(pack.dst)
                    #print(chemin[0], chemin[1])
                    #print(self.id_index(chemin[0]),self.id_index(chemin[1]))
                    route_time = self.route_time_matrix[self.id_index(chemin[0])][self.id_index(chemin[1])]
                    t_arrive = time + route_time
                    #print(chemin[1:])
                    #print(pack.dst)
                    self.events.put([t_arrive, pack, loc.ID, chemin[1:], "ARRIVED"])
        #print(1)

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
        df.to_csv('Tracking_info.csv', index=False)



if __name__ == '__main__':
    # 生成模拟数据
    data = gen.data_gen()

    # 导入数据
    sim = simulator(data)
    sim2 = simulator2(data)


    sim.pack_deal()
    sim2.pack_deal()


    i = 0
    j = 0
    for pack in sim.packages :
        if pack.category == 1:
            if sim2.tracking_info[pack.ID][-1][0] < sim.tracking_info[pack.ID][-1][0]:

                i =i+1
            if sim2.tracking_info[pack.ID][-1][0] > sim.tracking_info[pack.ID][-1][0]:
                j = j+1




    print(f"The number of slower packages is {i}.")
    print(f"The number of faster packages is {j}.")

