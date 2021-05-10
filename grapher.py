import plotly.graph_objects as go
from random import randint, random
import networkx as nx
import heapq
from networkx.generators.random_graphs import erdos_renyi_graph

from datetime import datetime

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0
    def empty(self):
        return not self._queue
    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1
    def get(self):
        return heapq.heappop(self._queue)[-1]


class Graph():
  
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[None for column in range(vertices)] for row in range(vertices)]

    def minDistance(self, dist, sptSet):
        min = float('inf')
        for v in range(self.V):

            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
  
        return min_index

    def dijkstra(self, src):
        st = datetime.now()
        dist = [float('inf')] * self.V
        prev = [None] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
  
        for cout in range(self.V):
            
            u = self.minDistance(dist, sptSet)

            sptSet[u] = True
            for v in range(self.V):
                if self.graph[u][v] is not None and sptSet[v] == False and \
                dist[v] > dist[u] + self.graph[u][v]:
                        dist[v] = dist[u] + self.graph[u][v]
                        prev[v] = u
        
        return dist, prev

    def bellman (self,src, edges):
        dist = [float('inf')] * self.V
        prev = [None] * self.V
        dist[src] = 0
        n_edges=len(edges)
        for i in range(1,self.V):
            for j in range(n_edges):
                u=edges[j][0]
                v=edges[j][1]
                weight=edges[j][2]
                if (dist[u] != float('inf') and dist[u] + weight < dist[v]):
                    dist[v] = dist[u] + weight
                    prev[v] = u
        return dist, prev
    
    def heuristic(self, x, y, g):
        x1,y1 = g.nodes[x]['pos']
        x2,y2 = g.nodes[y]['pos']
        h = ((x1-x2)**2 + (y1-y2)**2)**0.5
        return h
    
    def a_star_search(self, start, goal,g):
        frontier = PriorityQueue()
        frontier.push(start, 0)
        came_from = [None] * self.V
        cost_so_far = [float('inf')] * self.V
        came_from[start] = None
        cost_so_far[start] = 0

        while not frontier.empty():
            u = frontier.get()
            if u == goal:
                break
            for v in range(self.V):
                if self.graph[u][v] is not None  and cost_so_far[u] + self.graph[u][v] < cost_so_far[v]:
                    cost_so_far[v] = cost_so_far[u] + self.graph[u][v]
                    priority = cost_so_far[v] + self.heuristic(v, goal,g)
                    frontier.push(v, priority)
                    came_from[v] = u

        return cost_so_far, came_from

def ret_fig(n):
    # simulation
    import time
    
    N=[n]
    P=[0.5]

    timeList=[[None for i in range(0,len(P))] for j in range (0,len(N))]
    i=0
    goal = randint(1,n-1)
    for n in N:
        j=0
        for p in P:
            error=True
            while True:
                g = nx.random_geometric_graph(n, 0.2) # generating graph
                if nx.is_connected(g):
                    break
                else:
                    continue
            w_graph= [[None for i in range(0,n)] for j in range (0,n) ]
            # print(w_graph)
            edges=[]
            for samp in g.edges:
                x1,y1 = g.nodes[samp[0]]['pos']
                x2,y2 = g.nodes[samp[1]]['pos']
                dist = ((x1-x2)**2 + (y1-y2)**2)**0.5
                w_graph[samp[0]][samp[1]]=dist
                w_graph[samp[1]][samp[0]]=dist
                edges.append((samp[0],samp[1],dist))
                edges.append((samp[1],samp[0],dist))
            graph = Graph(n)
            graph.graph=w_graph
            dijkstra_start_time=time.time()
            a,b=graph.dijkstra(0)
            # print(a,b)
            dijkstra_end_time=time.time()
            bellman_end_time = 0
            bellman_start_time = 0
            if n < 301:
                bellman_start_time=time.time()
                a1,b1=graph.bellman(1,edges)
                # print(a)
                bellman_end_time=time.time()
            astar_start_time=time.time()
            a2,b2 = graph.a_star_search(0,goal,g)
            astar_end_time=time.time()
            timeList[i][j]=(dijkstra_end_time-dijkstra_start_time,bellman_end_time-bellman_start_time,astar_end_time-astar_start_time,n)
            j=j+1
            i=i+1
    if n>300:
        return None, timeList
        
    path = [goal]
    src = 0
    while True:
        dest = b[path[-1]]
        if dest is None:
            break
        else:
            path.append(dest)
    path = [i for i in path[::-1]]
    visible_edges = []
    gedges = list(g.edges)
    for i in range(len(path)-1):
        try:
            visible_edges.append(gedges.index((path[i],path[i+1])))
        except:
            visible_edges.append(gedges.index((path[i+1],path[i])))

    ve = set(visible_edges)
    G = g

    edge_x = []
    edge_y = []
    edge_x_ = []
    edge_y_ = []
    for ind, edge in enumerate(G.edges()):
        if ind in ve:
            x0, y0 = G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['pos']
            edge_x_.append(x0)
            edge_x_.append(x1)
            edge_x_.append(None)
            edge_y_.append(y0)
            edge_y_.append(y1)
            edge_y_.append(None)
        else:
            x0, y0 = G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.2, color='#888'),
        hoverinfo='none',
        mode='lines')

        

    path_trace = go.Scatter(
        x=edge_x_, y=edge_y_,
        line=dict(width=1, color='red'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Electric',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Distance From Source',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.nodes):
        node_text.append('#: '+str(node))
    carr = a[::]
    carr[0] = '#41fc03'
    carr[goal] = '#fc030b'
    node_trace.marker.color = carr
    node_trace.text = node_text


    fig = go.Figure(data=[edge_trace, node_trace, path_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig, timeList