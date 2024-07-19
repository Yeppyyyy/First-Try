import random
import numpy as np
import matplotlib.pyplot as plt
import uuid
from sklearn.cluster import KMeans
import pandas as pd

parameters = {
    "station_num": 25,
    "center_num": 5,
    "packet_num": 15000,
}


def data_gen():
    # Generate Stations
    fig, ax = plt.subplots()
    station_pos = []
    location =[]
    # properties are defined here: throughput/tick, time_delay, money_cost
    station_prop_candidates = [
        (10, 2, 0.5), (15, 2, 0.6), (20, 1, 0.8), (25, 1, 0.9)]
    station_prop = []
    for i in range(parameters["station_num"]):
        # Map size is defined here, which is 100*100
        station_pos.append((random.randint(0, 100), random.randint(0, 100)))
        station_prop.append(
            station_prop_candidates[random.randint(0, len(station_prop_candidates)-1)])
        location.append((f"s{i}", station_pos[i]))
        plt.text(station_pos[i][0], station_pos[i][1], f"s{i}", fontsize=9, ha='center', va='center', color='black')
    # Output Stations
    #print("Stations:")
    #for i in range(len(station_pos)):
        #print(f"s{i}", station_pos[i], station_prop[i])
        #location.append((f"s{i}", station_pos[i]))
    # Generate Centers by clustering
    kmeans = KMeans(n_clusters=parameters["center_num"])
    kmeans.fit(station_pos)
    station_labels = kmeans.predict(station_pos)
    center_pos = [(int(x[0]), int(x[1])) for x in kmeans.cluster_centers_]
    for i in range(len(center_pos)):
        while center_pos[i] in station_pos:
            # move slightly if center is overlapped with station
            # you can also use other methods to avoid this situation
            #print("Warning: Center moved")
            center_pos[i] = center_pos[i][0] + 1, center_pos[i][1] + 1
    # properties are defined here: throughput/tick, time_delay, money_cost
    center_prop_candidates = [
        (100, 2, 0.5), (150, 2, 0.5), (125, 1, 0.5), (175, 1, 0.5)]
    center_prop = []
    for i in range(parameters["center_num"]):
        center_prop.append(
            center_prop_candidates[random.randint(0, len(center_prop_candidates)-1)])
        location.append((f"c{i}", center_pos[i]))
    # Output Centers
    #print("Centers:")
    #for i in range(parameters["center_num"]):
        #print(f"c{i}", center_pos[i], center_prop[i])
        #location.append((f"c{i}", center_pos[i]))
        plt.scatter([center_pos[i][0]], [center_pos[i][1]], c='black', s=400, alpha=0.5)
        plt.text(center_pos[i][0], center_pos[i][1],f"c{i}", color='yellow',fontsize=12, ha='center', va='center')
    # Draw Stations and Centers

    plt.scatter([x[0] for x in station_pos], [x[1] for x in station_pos], c=station_labels, s=200, cmap='viridis',alpha=0.5)
    #plt.scatter([x[0] for x in center_pos], [x[1]
                #for x in center_pos], c='black', s=200, alpha=0.5)


    # Generate Edges
    edges = []
    #print("Edges (center to center):")      # Airlines
    for i in range(parameters["center_num"]):
        for j in range(parameters["center_num"]):
            if j > i:
                dist = np.linalg.norm(
                    np.array(center_pos[i]) - np.array(center_pos[j]))
                # src, dst, time_cost, money_cost
                # time_cost and money_cost are defined here
                edges.append((f"c{i}", f"c{j}", 0.25 * dist, 0.2 * dist))
                edges.append((f"c{j}", f"c{i}", 0.25 * dist, 0.2 * dist))
                plt.plot([center_pos[i][0], center_pos[j][0]], [
                         center_pos[i][1], center_pos[j][1]], 'r--')
                #print(edges[-2])
                #print(edges[-1])
    #print("Edges (center to station):")     # Highways
    for i in range(parameters["center_num"]):
        for j in range(parameters["station_num"]):
            if station_labels[j] == i:
                dist = np.linalg.norm(
                    np.array(center_pos[i]) - np.array(station_pos[j]))
                # time_cost and money_cost are defined here
                edges.append((f"c{i}", f"s{j}", 0.6 * dist, 0.12 * dist))
                edges.append((f"s{j}", f"c{i}", 0.6 * dist, 0.12 * dist))
                plt.plot([center_pos[i][0], station_pos[j][0]], [
                         center_pos[i][1], station_pos[j][1]], 'b--')
                #print(edges[-2])
                #print(edges[-1])
    #print("Edges (station to station):")    # Roads
    for i in range(parameters["station_num"]):
        for j in range(parameters["station_num"]):
            if i > j and (np.linalg.norm(np.array(station_pos[i]) - np.array(station_pos[j])) < 30):
                dist = np.linalg.norm(
                    np.array(station_pos[i]) - np.array(station_pos[j]))
                # time_cost and money_cost are defined here
                edges.append((f"s{i}", f"s{j}", 0.8 * dist, 0.07*dist))
                edges.append((f"s{j}", f"s{i}", 0.8 * dist, 0.07*dist))
                plt.plot([station_pos[i][0], station_pos[j][0]], [
                         station_pos[i][1], station_pos[j][1]], 'g--')
                #print(edges[-2])
                #print(edges[-1])
    #plt.show()

    # Generate Packets
    packets = []
    src_prob = np.random.random(parameters["station_num"])
    src_prob = src_prob / np.sum(src_prob)
    dst_prob = np.random.random(parameters["station_num"])
    dst_prob = dst_prob / np.sum(dst_prob)
    # Package categories are defined here: 0 for Regular, 1 for Express
    speed_prob = [0.7, 0.3]
    #print("Packets:")
    for i in range(parameters["packet_num"]):      # Number of packets
        src = np.random.choice(parameters["station_num"], p=src_prob)
        dst = np.random.choice(parameters["station_num"], p=dst_prob)
        while dst == src:
            dst = np.random.choice(parameters["station_num"], p=dst_prob)
        category = np.random.choice(2, p=speed_prob)
        # Create time of the package, during 12 time ticks(hours). Of course you can change it.
        create_time = np.random.random() * 12
        # create_time = 0.5
        packets.append([create_time, f"s{src}", f"s{dst}", category])
    # Sort packets by create time
    packets.sort(key=lambda x: x[0])
    # Output Packets
    #for packet in packets:
        #print(uuid.uuid4(), packet)

    llocation= dict(location)

    # 存到csv文件中
    new_prop = station_prop + center_prop
    df_prop = pd.DataFrame(new_prop, columns=['Column1', 'Column2', 'Column3'])
    df_matrix = pd.DataFrame(list(llocation.items()), columns=['Column1', 'Column2'])
    column_names = ['ID', 'Position', 'Throughput', 'Delay', 'Cost']
    df = pd.DataFrame(columns=column_names)
    df[['ID', 'Position']] = df_matrix
    df[['Throughput', 'Delay', 'Cost']] = df_prop
    df.to_csv('location.csv', index=False)

    column_names = ['Src', 'Dst', 'Time', 'Cost']
    df2 = pd.DataFrame(columns=column_names)
    df_edges = pd.DataFrame(edges, columns=['Column1', 'Column2', 'Column3', 'Column4'])
    df2[['Src', 'Dst', 'Time', 'Cost']] = df_edges
    df2.to_csv('routes.csv', index=False)

    column_names = ['ID', 'TimeCreated', 'Src', 'Dst', 'Category']
    df3 = pd.DataFrame(columns=column_names)
    for packet in packets:
        packet.insert(0, uuid.uuid4())
    df_packets = pd.DataFrame(packets, columns=['Column1', 'Column2', 'Column3', 'Column4', 'Column5'])
    df3[['ID', 'TimeCreated', 'Src', 'Dst', 'Category']] = df_packets
    df3.to_csv('packets.csv', index=False)

    return {
        "station_pos": station_pos,
        "station_prop": station_prop,
        "center_pos": center_pos,
        "center_prop": center_prop,
        "edges": edges,
        "packets": packets,
        "fig": fig,
        "ax": ax,
        "location":llocation,

    }
