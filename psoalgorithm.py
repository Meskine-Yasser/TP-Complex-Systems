# Task 1
# GUERRICHA MED. SABER & MESKINE YASSER
import random
import numpy as np
#from matplotlib import pyplot as plt
#from matplotlib import animation


# PSO Algorithm
def fitness_function(x, y):
    f1 = x ** 2 - y - 3
    return f1


def update_velocity(particle, velocity, pbest, gbest, w, c1, c2):
    # Initialise new velocity array
    num_particle = len(particle)
    new_velocity = np.array([0 for i in range(num_particle)])
    # Randomly generate r1, r2 and inertia weight from normal distribution
    r1 = random.randrange(0, 100, 1) / 100
    r2 = random.randrange(0, 100, 1) / 100
    # Calculate new velocity
    for i in range(num_particle):
        new_velocity[i] = w * velocity[i] + c1 * r1 * (pbest[i] - particle[i]) + c2 * r2 * (gbest[i] - particle[i])
    return new_velocity


def update_position(particle, velocity):
    # Move particles by adding velocity
    new_particle = (particle + velocity) % 31
    return new_particle


def pso_2d(population, dimension, position_min, position_max, generation, fitness_criterion, w, c1, c2):
    # Initialisation
    # Population
    particles = [[random.randint(position_min, position_max) for j in range(dimension)] for i in range(population)]
    # Particle's best position
    pbest_position = particles
    # Fitness
    pbest_fitness = [fitness_function(p[0], p[1]) for p in particles]
    # Index of the best particle
    gbest_index = np.argmax(pbest_fitness)
    # Global best particle position
    gbest_position = pbest_position[gbest_index]
    # Velocity (starting from 0 speed)
    velocity = [[0 for j in range(dimension)] for i in range(population)]

    # Loop for the number of generation
    for t in range(generation):
        # Stop if the average fitness value reached a predefined success criterion
        aver = 1 #np.average(pbest_fitness)
        if aver >= fitness_criterion:
            break
        else:
            for n in range(population):
                # Update the velocity of each particle
                velocity[n] = update_velocity(particles[n], velocity[n], pbest_position[n], gbest_position, w, c1, c2)
                # Move the particles to new position
                particles[n] = update_position(particles[n], velocity[n])
        # Calculate the fitness value
        pbest_fitness = [fitness_function(p[0], p[1]) for p in particles]
        # Find the index of the best particle
        gbest_index = np.argmax(pbest_fitness)
        # Update the position of the best particle
        gbest_position = pbest_position[gbest_index]
        print("G Best Position: " + str(gbest_position) + " and its value: " + str(max(pbest_fitness)))

    # Print the results
    print()
    print('Global Best Position: ', gbest_position)
    print('Best Fitness Value: ', max(pbest_fitness))
    print('Number of Generation: ', t)

    # # Plotting preparation
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='2d')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    #
    # x = np.linspace(position_min, position_max, 80)
    # y = np.linspace(position_min, position_max, 80)
    # X, Y = np.meshgrid(x, y)
    # Z = fitness_function(X, Y)
    # ax.plot_wireframe(X, Y, Z, color='r', linewidth=0.2)
    #
    # # Animation image placeholder
    # images = []
    #
    # # Add plot for each generation (within the generation for-loop)
    # image = ax.scatter3D([
    #     particles[n][0] for n in range(population)],
    #     [particles[n][1] for n in range(population)],
    #     [fitness_function(particles[n][0], particles[n][1]) for n in range(population)], c='b')
    # images.append([image])
    #
    # # Generate the animation image and save
    # animated_image = animation.ArtistAnimation(fig, images)
    # animated_image.save('./pso_simple.gif', writer='pillow')


if __name__ == '__main__':
    print("Started Optimization")
    population = 100
    #dimension = 2
    position_min = 0
    position_max = 31
    generation = 100
    fitness_criterion = 897
    c1 = c2 = 2
    w = 0.5
    pso_2d(population, 2, position_min, position_max, generation, fitness_criterion, w, c1, c2)
    print("Finished Optimization")
