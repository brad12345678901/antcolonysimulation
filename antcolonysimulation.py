import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle
import os
import random
import time
import pandas as pd
import csv
from datetime import datetime
from tqdm import tqdm

debugbestsolution = False #DEBUG FOR BCOMPARING BEST SOLUTION
debugsolutionprint = False
debugfinalprintroute = True
debugprintduringplottingedges = False
debugtraversal = False  #TRAVERSAL DEBUG
debugsaveprint = False  #PRINTS SAVE FILE
record = True  #RECORD DATA

class antcolonyoptimization():
    def __init__(self,
                 ants:int,
                 Q:int,
                 alpha1:float,beta1:float,rho:float,iterations:int, 
                 alpha2:float = 1.0,beta2:float = 1.0,
                 avgspeed=40,numoftrucks=5,wastecapacity=500,
                 dw=0.25,tw=0.25,ww=0.25,nw=0.25):
        #ACO PARAMETERS
        self.ants = ants
        self.Q = Q
        self.alpha = alpha1
        self.beta = beta1
        self.rho = rho
        self.iterations = iterations
        #TRUCK CHARACTERISTICS
        self.wastecapacity = wastecapacity
        self.numoftrucks = numoftrucks
        self.avgspeed = round(avgspeed * 1609.344,2)  #40MPH => 64373.8 Meters per Hour
        #GRAPH
        self.g = None
        #CONGESTION LEVEL FACTOR
        self.alphacon = alpha2
        self.betacon = beta2
        #MAX VALUES
        self.maxdistance = None
        self.maxtraveltime = None
        self.maxwastecollected = None
        self.maxnodes = None
        #WEIGHTS
        self.distancew = dw
        self.timew = tw
        self.wastecolw = ww
        self.nodesvisitw = nw
        
    def returnNeighbors(self, node_id:int, visited:list):
        neighbors = list(self.g.neighbors(node_id))
        unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in visited]
        return unvisited_neighbors
    
    def selectNextNode(self,current_node:int,neighbors:list,trucknum:int):
        total_probability = 0
        probability = []
        
        for neighbor in neighbors:
            edge_data = self.g.get_edge_data(current_node,neighbor)
            pheromone = edge_data.get('pheromone', [1])
            distance = edge_data.get('distance', 1)
            congestionlvl = edge_data.get('congestion', 1)
            
            heuristic = 1 / distance * (1 + congestionlvl)
            
            prob_factor = (pheromone[trucknum] ** self.alpha) * (heuristic ** self.beta)
            probability.append((neighbor, prob_factor))
            total_probability += prob_factor
        
        if total_probability == 0:
            return None
        
        probabilities = [(neighbor, prob_factor/total_probability) for neighbor, prob_factor in probability]
        
        next_node = random.choices([neighbor for neighbor, _ in probabilities],
                               weights=[prob_factor for _, prob_factor in probabilities])[0]
        
        return next_node
    
    def identifyedges(self, edges, ids:list):
        for i in range(0,len(ids)-1):
            get_edge = self.g.edges[ids[i]['nodeID'],ids[i+1]['nodeID']]
            edges.append([(ids[i]['nodeID'],ids[i+1]['nodeID']),get_edge])
            
    def pathtotal_speed(self,edges):
        time_per_lineseg = []
        totaltime = 0
        for i, v in edges:
            time = (v['distance'] / self.avgspeed) * (1 + v['congestion'])
            totaltime += (time*3600)
            time_per_lineseg.append({'edge':i,'traveltime':(time*3600)})
        return time_per_lineseg, totaltime
    
    def returnTotaldistance(self,edges):
        totaldistance = 0
        for i in edges:
            #print('DISTANCE ',i[1]['distance'])
            totaldistance += i[1]['distance']
        #print(totaldistance)
        return totaldistance
    
    def determineMaxValues(self,route):
        self.maxdistance = max([i['totaldistance'] for i in route])
        self.maxtraveltime = max([i['totaltime'] for i in route])
        self.maxwastecollected = max([i['totalwaste'] for i in route])
        self.maxnodes = max([len(i['visited']) for i in route])
        
    def evaluate_route(self, distance, time, waste_collected, nodes_visited):
        
        #LESS BETTER
        normalized_distance = self.normalize(distance, self.maxdistance)
        normalized_traveltime = self.normalize(time, self.maxtraveltime)
        #MORE BETTER
        normalized_wastecollected = 1 - self.normalize(waste_collected, self.maxwastecollected)
        normalized_nodes = 1- self.normalize(nodes_visited, self.maxnodes)
        
        weights = {
        'distance': self.distancew,  # Weight for distance (40%)
        'time': self.timew,      # Weight for time (40%)
        'waste': self.wastecolw,     # Weight for waste collected (10%)
        'nodes': self.nodesvisitw      # Weight for nodes visited (10%)
        }
        
        score = (weights['distance'] * normalized_distance +
                weights['time'] * normalized_traveltime +
                weights['waste'] * normalized_wastecollected +
                weights['nodes'] * normalized_nodes)
        
        return score
    
    def normalize(self, variable, max_value):
        return variable / max_value if max_value != 0 else 0
    
    def update_pheromones(self, route):
        for i, route in enumerate(route):
            for u, v in self.g.edges():
                self.g[u][v]['pheromone'][i] *= (1-self.rho)
            for u in route['edges']:
                self.g[u[0][0]][u[0][1]]['pheromone'][i] *= (1-self.rho)
                if route['totaltime'] > 0:
                    self.g[u[0][0]][u[0][1]]['pheromone'][i] += (self.Q / route['totaltime'])
            
    def reset_pheromones(self):
        print("RESET PHEROMONE VALUES")
        for i in range(0,self.numoftrucks):
            for u, v in self.g.edges():
                self.g[u][v]['pheromone'][i] = 1
                
    def compareBestSolutions(self, current_best, new_best):
        avg_score1 = sum(current_best)
        avg_score2 = sum(new_best)
        avg_score1 = avg_score1 / len(current_best)
        avg_score2 = avg_score2 / len(new_best)
        
        if debugbestsolution:
            print(f"AVG CURRENT: {avg_score1} AVG NEW: {avg_score2}")
        
        if avg_score1 > avg_score2:
            if debugbestsolution:
                print("NEW BEST IS BEST ROUTE")
            return True
        else:
            if debugbestsolution:
                print("New Best is not Best Route")
        
        return False
    
    def saveRecord(self, current_best_route, iteration, time, metricscore, filename):
        totaldistance = sum([i['totaldistance'] for i in current_best_route])
        maxtraveltime = max([i['totaltime'] for i in current_best_route])
        totalwastecollected = sum([i['totalwaste'] for i in current_best_route])
        maxnodes = sum([len(i['visited']) for i in current_best_route])
        avgscore = sum(metricscore)/len(metricscore)
        data = [{'Iteration':iteration,
                 'TotalDistance':totaldistance,
                 'MaxTime':maxtraveltime,
                 'TotalWasteCollected':totalwastecollected,
                 'SumofNodesVisited':maxnodes,
                 'AvgScore':avgscore,
                 'Time':time}]
        # Check if the file exists to determine whether to append or create a new file
        file_exists = os.path.exists(filename)
        with open(filename, 'a', newline="\n") as csvfile:
            fieldname = ['Iteration','TotalDistance','MaxTime','TotalWasteCollected','SumofNodesVisited','AvgScore','Time']
            writer = csv.DictWriter(csvfile,fieldnames=fieldname)
            if not file_exists:
                writer.writeheader()
            writer.writerows(data)
        if debugsaveprint:
            print(f"Data saved to {filename}")
        
    
    def createRecord(self, numtrucks):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = "data/routes"
        filename = f"routeTRUCKS{numtrucks}ITERATIONS{self.iterations}AL{self.alpha}BE{self.beta}Q{self.Q}RHO{self.rho}_W[{self.distancew},{self.timew},{self.wastecolw},{self.nodesvisitw}]_{timestamp}.csv"
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        return file_path
    
    def acomain(self):
        #START OF ACO
        start_node = [node for node, attr in self.g.nodes(data=True) if attr.get('subdiv') == 'START']
        
        if record:
            file = self.createRecord(self.numoftrucks)
            
        #print(start_node)
        #BEST ROUTE
        current_best = [None for i in range(0,self.numoftrucks)]
        current_bestscore = [None for i in range(0,self.numoftrucks)]
        current_bestnodes = 0
        for iteration in tqdm(range(0,self.iterations), desc="Processing features"):
            
            if record:
                start = time.time()
                
            visited1 = set()
            visited1.add(start_node[0])
            best_route = [None for i in range(0,self.numoftrucks)]
            best_metric_score = [None for i in range(0,self.numoftrucks)]
            bestrouteNodeRemains = 0
            for num in range(0,self.numoftrucks):
                solutions = []
                #CREATE SOLUTIONS FOR TRUCK # something
                for ant in range(0,self.ants):
                    route, edges, timesegments, totaldistance, totaltime, wastecollected, visited = self.ant_construct_sol(start_node[0],visited1.copy(),num)
                    if debugsolutionprint:
                        print(f"SOLUTION {ant+1} for TRUCK #{num+1}")
                    solutions.append({"solution":ant+1,
                                    'route':route,
                                    'edges':edges,
                                    'traveltimes':timesegments,
                                    'totaldistance':totaldistance,
                                    'totaltime':totaltime,
                                    'totalwaste':wastecollected,
                                    'visited':visited,
                                    'startid':start_node[0]})
                #DETERMINE MAX VALUES FROM SOLUTIONS
                self.determineMaxValues(solutions)
                #EVALUATE THE SCORE OF THE ROUTE
                metrics = []
                for i in solutions:
                    metrics.append((i['solution']-1, self.evaluate_route(i['totaldistance'],i['totaltime'],i['totalwaste'],len(i['visited']))))
                #DETERMINE THE BEST SOLUTION
                best_score = None
                for i in metrics:
                    if best_score is None:
                        best_score = i
                    if best_score[1] > i[1]:
                        best_score = i
                #SELECT BEST SOLUTION
                best_route[num] = solutions[best_score[0]]
                best_metric_score[num] = best_score[1]
                visited1 = visited1.union(best_route[num]['visited'])
                #GET REMAINING NODES
                if len(visited) == len(self.g.nodes()):
                    if debugsolutionprint:
                        print("ALL NODES VISITED")
                else:
                    if debugsolutionprint:
                        print("ALL NODES WAS NOT VISITED")
                    bestrouteNodeRemains = set(self.g.nodes()) - set(visited)
            if None in current_best or None in current_bestscore or current_bestnodes == 0:
                current_best = best_route
                current_bestscore = best_metric_score
                current_bestnodes = bestrouteNodeRemains
            else:
                if self.compareBestSolutions(current_bestscore,best_metric_score):
                    current_best = best_route
                    current_bestscore = best_metric_score
                    current_bestnodes = bestrouteNodeRemains
            self.update_pheromones(current_best)
            if record:
                end = time.time()
                self.saveRecord(current_best,iteration+1,(end-start), current_bestscore, file)
        
        return current_best, current_bestscore, current_bestnodes
                
    def ant_construct_sol(self,start_node,visited:list,trucknum:int):
        route = []
        edges = []
        truck_capacity = 0
        current_node = start_node
        route.append({'nodeID':current_node,
                        'pos':(self.g.nodes[current_node]['x'],self.g.nodes[current_node]['y']),
                        'WasteKg':self.g.nodes[current_node]['wastekg']})
        while True:
                next_node = aco.selectNextNode(current_node, aco.returnNeighbors(current_node,visited),trucknum)
                wastekgnext = self.g.nodes[next_node].get('wastekg')
                if debugtraversal:
                    print(f'CURRENT : {current_node} NEXT: {next_node} NEXT NODE WASTEKG: {wastekgnext} CAPACITY: [{self.wastecapacity - truck_capacity}]/[{self.wastecapacity}]')
                truck_capacity += wastekgnext
                if (self.wastecapacity - truck_capacity) < 0 or next_node == None:
                    truck_capacity -= wastekgnext
                    if debugtraversal:
                        print(f'trash cannot be stored anymore TOTAL WASTE COLLECTED: {truck_capacity} FREE SPACE: {self.wastecapacity-truck_capacity}')
                    route.append({'nodeID':start_node,
                                  'pos':(self.g.nodes[start_node]['x'],self.g.nodes[start_node]['y']),
                                  'WasteKg':self.g.nodes[start_node]['wastekg']})
                    self.identifyedges(edges, route)
                    totaldistance = self.returnTotaldistance(edges)
                    timesegments, totaltime = self.pathtotal_speed(edges)
                    return route, edges, timesegments, totaldistance, totaltime, truck_capacity, visited
                #append ID
                route.append({'nodeID':next_node,
                              'pos':(self.g.nodes[next_node]['x'],self.g.nodes[next_node]['y']),
                              'WasteKg':self.g.nodes[next_node]['wastekg']})
                visited.add(next_node)
                current_node = next_node
            
    def plotnodesedges(self, routes, remainnodes, starting_point):
        outputg = nx.DiGraph()
        colors = ['blue', 'violet', 'green', 'yellow', 'pink','brown','gray']
        node_colors = []
        edge_colors = []
        edge_styls = []
        print("PLOTTING")
        plt.figure(figsize = (8,6))
        # Step 1: Add nodes from remainnodes to outputg with their positions
        for node in remainnodes:
            if node in self.g.nodes:
                pos = self.g.nodes[node].get('pos', None)  # Retrieve position from self.g
                if pos:
                    outputg.add_node(node, pos=pos)
                    node_colors.append('cyan')  # Default color for remaining nodes          
        for i, j in enumerate(routes):
            for nodes in j['route']:
                if not nodes['nodeID'] in outputg.nodes():
                    outputg.add_node(nodes['nodeID'], pos=nodes['pos'])
                    if nodes['nodeID'] == starting_point:
                        node_colors.append('red')
                    else:
                        node_colors.append(colors[i%len(colors)])   
            print("END NODE")
            #pos=nx.spring_layout(outputg)
            pos = nx.get_node_attributes(outputg, 'pos')
            nx.draw_networkx_nodes(outputg, pos, node_size=50, node_color=node_colors)
            for edge in j['edges']:
                outputg.add_edge(edge[0][0], edge[0][1], **edge[1])
                if edge[0][0] == starting_point:
                    edge_styls.append('dashed')
                elif edge[0][1] == starting_point:
                    edge_styls.append((0,(1,1)))
                else:
                    edge_styls.append('solid')
                edge_colors.append(colors[i % len(colors)])
            print("END EDGE")
            #pos = nx.random_layout(outputg)
            nx.draw_networkx_edges(outputg, pos, width=1, edge_color=edge_colors,style=edge_styls, arrows=False)
            break
        plt.title("Routes and Remaining Nodes")
        plt.axis('off')
        plt.show()
            
    def initializeedges(self, nodes, g):
        max_distance = max(point1.distance(point2) for i, point1 in enumerate(nodes.geometry) for j, point2 in enumerate(nodes.geometry) if i != j)
        #max_distance = max_distance * 111139
        for row_num1, row_att1 in nodes.iterrows():
            point1 = row_att1.geometry
            for row_num2, row_att2 in nodes.iterrows():
                if row_num1 != row_num2:
                    point2 = row_att2.geometry
                    # Calculate distance between points
                    dist = point1.distance(point2)
                    #dist_in_m = dist * 111139
                    congestion_level = (self.alphacon * (dist/max_distance)) + (self.betacon*(random.uniform(0,1)))
                    g.add_edge(row_num1, row_num2, distance=dist, pheromone=1, congestion = congestion_level)
                    
    def initializeIndividualPheromones(self):
        pheromonelist = [1 for _ in range(self.numoftrucks)]
        for u,v in self.g.edges():
            self.g[u][v]['pheromone'] = pheromonelist
                    
    def setGraph(self,g):
        self.g = g.copy()
        
    def checkData(self):
        for node, attribute in self.g.nodes(data=True):
            print(f"Node {node}, ATTR: {attribute}")
        for node1, node2, attribute in self.g.edges(data=True):
            print(f"Edge ({node1},{node2}), ATTR: {attribute}")
                
def checkifgraphexist(path):
    return os.path.exists(path)
    
def loadgraph(gpath):
    return nx.read_graphml(gpath)

                
if __name__ == '__main__':
    
    pathfiletoshp = r"data\graph\collectionpoints.shp" #THIS IS THE SHP FILE FOR THE NODES
    pathofgraphml = 'data/graph/collectionpoints.graphml' #GRAPHML FILE WHERE EDGES AND NODES ARE SAVED
    
    #CREATE GRAPH
    g = nx.Graph()
    #INITIALIZE ACO
    #WEIGHTS {distance,time,wastecollected,nodesvisited}
    weights = [0.4,0.4,0.2,0.1] #DISTANCE WEIGHT, TIME WEIGHT, WASTE COLLECTED WEIGHT, NODES VISITED WEIGHT
    aco = antcolonyoptimization(ants=10,Q=1.3,alpha1=1,beta1=2,rho=0.15,iterations = 1000,
                                dw=weights[0],tw=weights[1],ww=weights[2],nw=weights[3])
    os.makedirs('data/graph', exist_ok=True)
    #CHECK GRAPH FILE -> CREATE A GRAPH
    if not checkifgraphexist(pathofgraphml):
        print("CREATING GRAPH")
        nodes = gpd.read_file(pathfiletoshp)
        #total_waste = 0
        for idx, row in nodes.iterrows():
            x = row.geometry.x
            y = row.geometry.y
            attrs = row.to_dict()
            if 'geometry' in attrs:
                del attrs['geometry']
            if attrs['subdiv'] == None:
                attrs['subdiv'] = 'placeholder'
            g.add_node(idx, x=x, y=y, **attrs)
            #total_waste += row['wastekg']
        aco.initializeedges(nodes,g)    #PLOT ALL EDGES
        
        nx.write_graphml(g, 'data/graph/collectionpoints.graphml')
        
    #IF THE GRAPH EXIST -> LOAD THE GRAPH
    else:
        print('loaded')
        g = loadgraph(pathofgraphml)
        
    aco.setGraph(g)     #SET THE GRAPH FROM ACO
    aco.initializeIndividualPheromones()
    aco.reset_pheromones()
    
    route, scores, remainnodes = aco.acomain()
    
    if debugfinalprintroute:
        for i, j in enumerate(zip(route,scores)):
            if not j[0] is None:
                print(j[0]['edges'])
                print(f"TRUCK NUM: {i+1} SOLUTION: {j[0]['solution']}, TOTALDISTANCE {round(j[0]['totaldistance'],2)}m, TOTALTIME: {round(j[0]['totaltime'], 2)}s, WASTECOLLECTED: {j[0]['totalwaste']}kg", end=" ||| ")
                print(f"SOLUTION: {j[0]['solution']}, METRICSCORE: {j[1]}")
        print('\n\nREMANINING NODES: ', remainnodes)
    
    aco.plotnodesedges(route, remainnodes, route[0]['startid'])
    

    
            
    
    
        
