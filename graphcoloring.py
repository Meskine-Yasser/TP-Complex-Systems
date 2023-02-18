#Task 3
# GUERRICHA MED. SABER & MESKINE YASSER
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path

# For the plot
Gen = np.array([])
Fit = np.array([])

t = Path("Myciel3.txt").read_text()  # open the file, read the file content and return it into a string variable and close the file
num = [[int(n) for n in l.split()] for l in t.splitlines()]  # turn the string into a matrix of integers (ex: [ [11, 20], [1, 2], [1, 4]])
colors = Path("colors.txt").read_text()
colors = colors.splitlines()
number_of_colors = 4


def toAdjMatrice():  # turn the int matrix to an Adjacency Matrix
    AdjM = np.zeros((num[0][0], num[0][0]))  # create an empty matrix with the height of the nodes number
    for n in num[1:]:
        AdjM[n[0] - 1][n[1] - 1] = 1  # turn each edge into a 1
        AdjM[n[1] - 1][n[0] - 1] = 1
    return AdjM


def MtoGraph(M):  # turn the int matrix to a colorless graph
    G = nx.Graph()
    for i in range(len(M)):
        G.add_node(i + 1, color="#000000")
        for j in range(len(M[i])):
            if M[i][j] == 1:
                G.add_edge(i + 1, j + 1)
    return G


def ColorTheGraph(G, individual):
    for n in G:  # iterating through the nodes of the fittest individual and color them
        G.nodes[n]["color"] = colors[individual[n - 1]]
    return G


graph = toAdjMatrice()
netGraph = MtoGraph(graph)
n = len(graph)
for l in graph:
    print(l)


def create_individual():
    individual = []
    #random individual x with random set of colors between 0-3
    for i in range(n):
        individual.append(random.randint(0, number_of_colors - 1))
    return individual


'''Create Population'''
population_size = 50
generation = 0
population = []
#population of individuals
for i in range(population_size):
    x = create_individual()
    population.append(x)

#check the fitness of the individual ( number of conflicts ) the best is 0
def fitness(graph, individual):
    fitness = 0
    for i in range(n):
        for j in range(i, n):
            if individual[i] == individual[j] and graph[i][j] == 1:
                fitness += 1
    return fitness


#take 2 individuals and create 2 new individuals from them
#with a 100% possibility to happen
#parent1 --> [00000000]
#parent2 --> [11111111]
#random position from the array to split from ex --> 5
#child1 --> [0000*1111111]
#child2 --> [1111*0000000]
def crossover(parent1, parent2):
    probability = 1
    check = random.uniform(0, 1)
    if check <= probability:
        position = random.randint(0, n - 1)
        child1 = []
        child2 = []
        for i in range(position):
            child1.append(parent1[i])
            child2.append(parent2[i])
        for i in range(position, n):
            child1.append(parent2[i])
            child2.append(parent1[i])
    return child1, child2

#change a color from the vector of an individual with a 20% possibility to happen
def mutation(individual):
    probability = 0.2
    check = random.uniform(0, 1)
    if check <= probability:
        position = random.randint(0, n - 1)
        individual[position] = random.randint(1, number_of_colors - 1)
    return individual



#select the best from 2 individuals and add it to the next generation population
def tournament_selection(population):
    newPop = []
    for j in range(2):
        random.shuffle(population)
        for i in range(0, population_size - 1, 2):
            if fitness(graph, population[i]) < fitness(graph, population[i + 1]):
                newPop.append(population[i])
            else:
                newPop.append(population[i + 1])
    return newPop

#set initial parameters
best_fitness = fitness(graph, population[0])
bestX = population[0]
gen = 0
#timer to calculate the time needed to get this valid graph
start = time.time()
#main loop to create generations
while gen < 100:
    gen += 1
    population = tournament_selection(population)
    new_population = []
    random.shuffle(population)
    for i in range(0, population_size - 1, 2):
        child1, child2 = crossover(population[i], population[i + 1])
        new_population.append(child1)
        new_population.append(child2)
    for x in new_population:
        if gen < 100:
            x = mutation(x)
    population = new_population
    best_fitness = fitness(graph, population[0])
    bestX = population[0]
    for x in population:
        if fitness(graph, x) < best_fitness:
            best_fitness = fitness(graph, x)
            bestX = x
    end = time.time()
    print("Generation: ", gen, "Best_Fitness: ", best_fitness, "Individual: ", bestX)
    Gen = np.append(Gen, gen)
    Fit = np.append(Fit, best_fitness)
    if best_fitness == 0:
        exT = (end - start) * 10 ** 3
        print(f"This is a valid graph, time needed to be executed : {exT:.03f} ms")
        break

#if we get a valid graph, we show it with a plot of all the gens and their fitness
if best_fitness == 0:
    print("Graph is ", number_of_colors, " colorable")
    plt.plot(Gen, Fit)
    plt.figure(1)
    plt.xlabel("generation")
    plt.ylabel("best-fitness")
    ColorTheGraph(netGraph, bestX)
    colors = [node[1]['color'] for node in netGraph.nodes(data=True)]
    plt.figure(2)
    nx.draw(netGraph, font_size=30, node_size=1500, width=3, font_color="white", node_color=colors, with_labels=True)
    plt.tight_layout()
    plt.axis("off")
    plt.show()
else:
    print("This graph isn't ", number_of_colors, " colorable")
