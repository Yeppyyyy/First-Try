import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

'''
用法：import以后ani.animate(sim, idd, data)
'''


def animate(sim,idd):
# 假设这是输入数据格式
    package_statuses = sim.tracking_info[idd][4:]

# 将位置映射到二维坐标
    location_to_coord = sim.locations # location的坐标
    fig_bg,ax = sim.fig, sim.ax
# 初始化图形和轴
    #ax = fig_bg.add_subplot(1,1,1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    text = ax.text(0.34, 1.08, "", transform=ax.transAxes, fontsize=12, ha='left', va='center')


# 定义绘制初始状态的函数
    def init():
        pass
# 更新函数
    def update_plot(time):
        deltat =0.2
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        text.set_text(f"Current time: {time:.2f}")
    # 如果找的到记录
        op = False
        for status in package_statuses:
            nextcoord = location_to_coord[sim.id_index(status[1])].pos
            nexttime = status[0]
            if (time - status[0]) <= deltat and (time - status[0]) > 0:
                pretime = status[0]
                precoord = location_to_coord[sim.id_index(status[1])].pos
                etat = status[2]
                op = True
            elif time < status[0]:
                break
    # 如果找得到记录
        if op == True:
            if etat == 'PROCESSING':
                color = 'blue'
                ax.plot([precoord[0]], [precoord[1]], 'o', color=color)
            elif etat == 'SENT':
                color = 'pink'
                if nexttime != pretime:
                    time_diff = nexttime - pretime
                    time_passed = time - pretime
                    ratio = time_passed / time_diff
                    coord = (
                        precoord[0] + ratio * (nextcoord[0] - precoord[0]),
                        precoord[1] + ratio * (nextcoord[1] - precoord[1]))
                    ax.plot([coord[0]], [coord[1]], 'o', color=color)
            else:
                color = 'red'
                ax.plot([precoord[0]], [precoord[1]], 'o', color=color)

        else:  # 找不到则重新搜索记录
            prestatuses = [s for s in package_statuses if s[0] <= time]
            if prestatuses!=[]:
                prestatus = prestatuses[-1]
                pretime2 = prestatus[0]
                precoord2 = location_to_coord[sim.id_index(prestatus[1])].pos
                if prestatus[2] == 'ARRIVED':
                    color = 'red'  # 代表wait for processing
                    ax.plot([precoord2[0]], [precoord2[1]], 'o', color=color)
                elif prestatus[2] == 'PROCESSING':
                    color = 'blue'  # 代表processing
                    ax.plot([precoord2[0]], [precoord2[1]], 'o', color=color)
                elif prestatus[2] == 'SENT':
                    if nexttime != pretime2:
                        time_diff = nexttime - pretime2
                        time_passed = time - pretime2
                        ratio = time_passed / time_diff
                        coord = (
                            precoord2[0] + ratio * (nextcoord[0] - precoord2[0]),
                            precoord2[1] + ratio * (nextcoord[1] - precoord2[1]))
                        color = 'pink'
                        ax.plot([coord[0]], [coord[1]], 'o', color=color)
        if time >= max(status[0]for status in package_statuses):
            ani.event_source.stop()
# 设置动画对象
    ani = animation.FuncAnimation(fig_bg, update_plot, frames=np.arange(0.2, max(status[0] for status in package_statuses)+0.2, 0.2), init_func=init, interval=20)

# 显示图形
    plt.show()
