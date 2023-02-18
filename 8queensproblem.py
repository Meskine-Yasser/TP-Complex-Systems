#Task 2
# GUERRICHA MED. SABER & MESKINE YASSER
from operator import indexOf
import random
import time
from matplotlib import pyplot as plt


# Making random chromosomes
def random_chromosome(size):
    return [random.randint(0, size - 1) for _ in range(size)]


# Calculating fitness
def fitness(chromosome, maxFitness):
    # Calculate the number of Occurrences in the chromosome
    # Horizontal is sufficient because the the filling od the solution will always guarantee there is no vertical collisions
    horizontal_collisions = (sum([chromosome.count(queen) - 1 for queen in chromosome]) / 2)

    n = len(chromosome)
    # Fill the diagonals with 0 to start calculating number of queens found
    left_diagonal = [0] * (2 * n - 1)
    right_diagonal = [0] * (2 * n - 1)
    # Calculate the number of queens found in each diagonal
    for i in range(n):
        left_diagonal[i + chromosome[i] - 1] += 1
        right_diagonal[len(chromosome) - i + chromosome[i] - 2] += 1

    # Calculate how many collisions are there in the all diagonals
    pass
    # Left Diagonal
    # [       0\0\0\0\0\0\0\0\]
    # [      0\0\0\0\0\0\0\0\]
    # [     0\0\0\0\0\0\0\0\]
    # [    0\0\0\0\0\0\0\0\]
    # [   0\0\0\0\0\0\0\0\]
    # [  0\0\0\0\0\0\0\0\]
    # [ 0\0\0\0\0\0\0\0\]
    # [0\0\0\0\0\0\0\0\]
    # Right Diagonal
    # [0/0/0/0/0/0/0/0/]
    # [ /0/0/0/0/0/0/0/0/]
    # [  /0/0/0/0/0/0/0/0/]
    # [   /0/0/0/0/0/0/0/0/]
    # [    /0/0/0/0/0/0/0/0/]
    # [     /0/0/0/0/0/0/0/0/]
    # [      /0/0/0/0/0/0/0/0/]
    # [       /0/0/0/0/0/0/0/0/]
    diagonal_collisions = 0
    for i in range(2 * n - 1):
        counter = 0
        if left_diagonal[i] > 1:
            counter += left_diagonal[i] - 1
        if right_diagonal[i] > 1:
            counter += right_diagonal[i] - 1
        diagonal_collisions += counter

    # 28-(2+3)=23
    return int(maxFitness - (horizontal_collisions + diagonal_collisions))


# Doing cross_over between two chromosomes
def crossover(x, y):
    n = len(x)
    child = [0] * n
    # for i in range(n):
    #     c = random.randint(0, 1)
    #     if c < 1:
    #         child[i] = x[i]
    #     else:
    #         child[i] = y[i]
    for i in range(n):
        if i % n >= n / 2:
            child[i] = y[i]
        else:
            child[i] = x[i]
    return child


# Randomly changing the value of a random index of a chromosome
def mutate(x):
    n = len(x)
    c = random.randint(0, n - 1)
    m = random.randint(0, n - 1)
    x[c] = m
    return x


# Calculating probability of the chromosome being close or far from the best fitness
def probability(chromosome, maxFitness):
    return fitness(chromosome, maxFitness) / maxFitness


# Roulette-wheel selection
def random_pick(population, probabilities):
    populationWithProbabilty = zip(population, probabilities)
    total = sum(w for c, w in populationWithProbabilty)
    r = random.uniform(0, total)
    upto = 0
    for c, w in zip(population, probabilities):
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"


# Genetic algorithm
def genetic_queen(population, maxFitness):
    user_mutation_probability = 0.5
    user_crossover_probability = 1
    new_population = []
    sorted_population = []
    probabilities = []
    # Calculate The fitness of the population
    for n in population:
        f = fitness(n, maxFitness)
        probabilities.append(f / maxFitness)
        sorted_population.append([f, n])

    # Sort The population from best to worst
    sorted_population.sort(reverse=True)

    # Elitism
    new_population.append(sorted_population[0][1])  # the best gen
    new_population.append(sorted_population[-1][1])  # the worst gen
    elite1 = sorted_population[0][1]
    elite2 = sorted_population[1][1]

    for i in range(len(population) - 2):
        # Roulette Selection
        chromosome_1 = random_pick(population, probabilities)
        chromosome_2 = random_pick(population, probabilities)

        # Best 2 from each Gen
        # chromosome_1 = sorted_population[0][1]
        # chromosome_2 = sorted_population[1][1]

        # Creating two new chromosomes from 2 chromosomes
        crossover_probability = random.randrange(0, 100, 1) / 100
        if crossover_probability <= user_crossover_probability:
            child = crossover(chromosome_1, chromosome_2)

        # Mutation
        mutation_probability = random.randrange(0, 100, 1) / 100
        if mutation_probability <= user_mutation_probability:
            child = mutate(child)

        new_population.append(child)
        if fitness(child, maxFitness) == maxFitness:
            break
    return new_population


# prints given chromosome
def print_chromosome(chrom, maxFitness):
    print(
        "Chromosome = {},  Fitness = {}".format(str(chrom), fitness(chrom, maxFitness))
    )


# prints given chromosome board
def print_board(chrom):
    board = []

    for x in range(nq):
        board.append(["x"] * nq)

    for i in range(nq):
        board[chrom[i]][i] = "Q"

    def print_board(board):
        for row in board:
            print(" ".join(row))

    print()
    print_board(board)


def plot_result(generations_axis, fitness_values_axis, chrom):
    plt.figure(1)
    plt.plot(generations_axis, fitness_values_axis)

    # Add labels
    plt.xlabel('Generations')
    plt.ylabel('Fitness Value')

    # Show plot
    # draw_board()
    plt.figure(2)
    board = [[0] * 8 for _ in range(8)]
    for i in range(nq):
        board[chrom[i]][i] = 1
    plt.imshow(board, cmap='binary')
    plt.show()


def main():
    generations_axis = []
    fitness_values_axis = []
    print("Give the population number: ")
    # POPULATION_SIZE = int(input())
    POPULATION_SIZE = 100
    generation = 0
    # nq = int(input(print("Give the number of queens to solve for NxN: ")))
    # While the solution is not found
    start = time.time()
    while True:

        #nq = 8
        if nq == 0:
            break
        # Calculate the fitness for the given input size, by default is 28
        maxFitness = (nq * (nq - 1)) / 2  # 8*7/2 = 28
        # Give a Random Population
        population = [random_chromosome(nq) for _ in range(POPULATION_SIZE)]
        print(population)

        # generation = 1
        while (
                # Calculate the fitness value of the initial
                not maxFitness in [fitness(chrom, maxFitness) for chrom in population]
                and generation < 100
        ):

            population = genetic_queen(population, maxFitness)
            if generation:
                generations_axis.append(generation)
                fitness_values_axis.append(max([fitness(n, maxFitness) for n in population]))
                print("=== Generation {} ===".format(generation))
                print(
                    "Maximum Fitness = {}".format(
                        max([fitness(n, maxFitness) for n in population])
                    )
                )
                print("Best of this Gen: ")
                fitnessOfChromosomes = [fitness(chrom, maxFitness) for chrom in population]
                bestChromosomes = population[indexOf(fitnessOfChromosomes, max(fitnessOfChromosomes))]
                print_chromosome(bestChromosomes, maxFitness)

            generation += 1

        fitnessOfChromosomes = [fitness(chrom, maxFitness) for chrom in population]

        bestChromosomes = population[
            indexOf(fitnessOfChromosomes, max(fitnessOfChromosomes))
        ]

        if maxFitness in fitnessOfChromosomes:
            end = time.time()
            endd = (end - start) * 10 ** 3
            print(f"time to solution: {endd:.03f} ms")
            print("\nSolved in Generation {}!".format(generation - 1))

            print_chromosome(bestChromosomes, maxFitness)

            print_board(bestChromosomes)
            plot_result(generations_axis, fitness_values_axis, bestChromosomes)
            print("Want to continue ?")
            yes = int(input())
            if yes:
                main()
            break
            # time.sleep(2)

        else:
            endfail = time.time()
            endfaild = (endfail - start) * 10 ** 3
            print(
                "\nUnfortunately, we could't find the answer until generation {}. The best answer that the algorithm found was:".format(
                    generation - 1
                )
            )
            print(f"time to last gen: {endfaild:.03f} ms")
            print_board(bestChromosomes)
            plot_result(generations_axis, fitness_values_axis, bestChromosomes)
            main()
            break


if __name__ == "__main__":
    nq = 8
    main()
