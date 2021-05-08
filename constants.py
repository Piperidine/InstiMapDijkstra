edges = [
    [1,5],
    [5,4],
    [4,2],
    [2,3],
    [3,37],
    [37,18],
    [18,17],
    [17,19],
    [19,20],
    [20,22],
    [22,23],
    [23,24],
    [20,21],
    [19,25],
    [25,27],
    [27,26],
    [26,28],
    [28,29],
    [29,30],
    [30,31],
    [31,32],
    [31,33],
    [33,34],
    [34,35],
    [35,40],
    [40,41],
    [41,42],
    [42,43],
    [43,44],
    [44,45],
    [2,45],
    [44,46],
    [37,36],
    [36,30],
    [36,38],
    [38,35],
    [38,39],
    [17,16],
    [16,14],
    [14,15],
    [14,13],
    [13,12],
    [12,9],
    [9,10],
    [10,11],
    [7,12],
    [8,9],
    [6,8],
    [8,7],
    [4,7],
    [5,6],
]

class Graph():
  
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)] 
                    for row in range(vertices)]

    def minDistance(self, dist, sptSet):
        min = float('inf')
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
  
        return min_index
    def dijkstra(self, src):
  
        dist = [float('inf')] * self.V
        prev = [None] * self.V
        dist[src] = 0
        sptSet = [False] * self.V
  
        for cout in range(self.V):
            u = self.minDistance(dist, sptSet)

            sptSet[u] = True
            for v in range(self.V):
                if self.graph[u][v] > 0 and sptSet[v] == False and \
                dist[v] > dist[u] + self.graph[u][v]:
                        dist[v] = dist[u] + self.graph[u][v]
                        prev[v] = u
        return dist, prev

def distance(lat1,lon1,lat2,lon2):
    R = 6373.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance * 1000