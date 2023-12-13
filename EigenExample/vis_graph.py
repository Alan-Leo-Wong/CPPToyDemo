"""
Author: Alan Wang leiw1006@gmail.com
Date: 2023-12-13 20:37:55
LastEditors: Alan Wang leiw1006@gmail.com
LastEditTime: 2023-12-13 20:48:43
FilePath: \EigenExample\vis_graph.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
import os
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io

# 获取当前目录下名为"graph"的子目录的路径
graph_dir = "graph"
graph_path = os.path.join(os.getcwd(), graph_dir)
# print("path: ", graph_path)

# 获取当前目录下所有后缀为.mtx的文件
mtx_files = [file for file in os.listdir(graph_path) if file.endswith(".mtx")]

# 遍历每个.mtx文件并生成可视化图
for mtx_file in mtx_files:
    print(f"Processing file: {mtx_file}")

    # 构建文件路径
    mtx_path = os.path.join(graph_path, mtx_file)

    # 从文件加载图数据
    G_data = scipy.io.mmread(mtx_path)

    # 将稀疏矩阵转换为 networkx 图
    G = nx.Graph(G_data)

    # 绘制图
    pos = nx.spring_layout(G)  # 使用布局算法布置节点
    nx.draw(
        G,
        pos,
        with_labels=True,
        font_weight="bold",
        node_size=50,
        node_color="skyblue",
        font_size=1,
        font_color="black",
    )

    # # 绘制边的权重
    # edge_labels = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # 显示图
    plt.title(f"Graph Visualization - {mtx_file}")
    plt.show()
