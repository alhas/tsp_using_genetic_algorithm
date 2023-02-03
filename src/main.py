from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from typing import List
import reader
import random
import numpy
import math

CITY_COORDINATES = reader.cities_in_array("../data/berlin52.tsp")
TOTAL_CHROMOSOME = len(CITY_COORDINATES) - 1

POPULATION_SIZE = 200
MAX_GENERATION = 100
MUTATION_RATE = 0.2
WEAKNESS_THRESHOLD = 26000 # Berlin11 = 5_700, Berlin52 = 26_000, kroA100= 156_000, kroA150= 240_000



class Genome():  # City class for our case.
    def __init__(self):  # constructor
        self.chromosome = []  # we have value for cities which is their values.
        self.fitness = 0  # fitness value is 0 for every city

    # returns string representation of constructor class of variables.
    def __str__(self):
        return f"Chromosome: {self.chromosome} Fitness: {self.fitness}\n"

    def __repr__(self):  # returns the object representation in string format.
        return str(self)


# the function create a genome and wait for return Genome object.
def create_genome() -> Genome:
    genome = Genome()  # we create genome object from Genome Class

    # genome chromosome will be the shuffeled random chromosomes.
    genome.chromosome = random.sample(range(1, TOTAL_CHROMOSOME + 1), TOTAL_CHROMOSOME)
    # ***********for this chromosome fitness value will be return from eval_chromosome.
    genome.fitness = eval_chromosome(genome.chromosome)
    return genome  # return genome object.


def distance(a, b) -> float:  # distance function takes two argument and return float
    # measure distance from two chromosome in our case two citiy Euclidean distance is will calculate the measurement.
    dis = math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))
    # return distance variable with roundup two decimal after point.
    return numpy.round(dis, 2)


# parameter is genome type of List function aspect Genome return value.
def get_fittest_genome(genomes: List[Genome]) -> Genome:
    # create genome_fitness list from created object genome.fitness for every genome in genomes parameter.
    genome_fitness = [genome.fitness for genome in genomes]
    # return genomes list from created genome_fitness.index of minimum genome_fitness
    return genomes[genome_fitness.index(min(genome_fitness))]  

# takes the chromosome argument as type of List and int value return float
def eval_chromosome(chromosome: List[int]) -> float:
    # fill with 0 the arr lengh of chromosome + 2
    arr = [0] * (len(chromosome) + 2)
    # add chromosome argument in to list between first and last value
    arr[1:-1] = chromosome

    fitness = 0
    for i in range(len(arr) - 1):  # for loop lenght of arr list
        p1 = CITY_COORDINATES[arr[i]]  # p1 is City coordinates arr i value
        p2 = CITY_COORDINATES[arr[i + 1]]  # p2 is next value of p1 index
        # fitness is return value from distance function.
        fitness += distance(p1, p2)
    # ***********return fitness value with 2 decimals after point.
    return numpy.round(fitness, 2)


# takes population argument type of List Genome object, and k int return List of Genome object.
def tournament_selection(population: List[Genome], k: int) -> List[Genome]:
    # select random parent from population created
    selected_genomes = random.sample(population, k)
    # return selected the fittest genome
    return get_fittest_genome(selected_genomes)


# take parents as a List Object return should be Genome object.
def one_point_crossover(parents: List[Genome]) -> Genome:
    # create a list fill of total lenght of chromosomes
    child_chro = [None] * TOTAL_CHROMOSOME

    # Select a random number between 2 and 6
    subset_length = random.randrange(2, 6)
    # Select random range from 0 to total chromosome - subset_lenght to randomize.
    crossover_point = random.randrange(0, TOTAL_CHROMOSOME - subset_length)

    '''
    child_chromosome slided by the crosover_point and crossover_point + 
    subset_length this values are asign from parent[0].chromosome's
    crossover_point:coossover_point + subset_length
    '''

    child_chro[crossover_point:crossover_point + subset_length] = parents[0].chromosome[crossover_point:crossover_point + subset_length]

    # j and k is equal to crossover_point + subset_length
    j, k = crossover_point + subset_length, crossover_point + subset_length

    # while None in child_chromosome list
    while None in child_chro:
        # If second element of parents 1 chromosome k not in child_chromosome
        if parents[1].chromosome[k] not in child_chro:
            # child_chromosome element of j assigned from parents[1].chromosome[k] value
            child_chro[j] = parents[1].chromosome[k]
            # the next j value equal to j + 1 if j isn't equal to TOTAL_CHROMOSOME -1 else j is 0
            j = j + 1 if (j != TOTAL_CHROMOSOME - 1) else 0
        # k is equal to k+1 if k is not equal to TOTAL _CHROMOSOME -1 else k = 0
        k = k + 1 if (k != TOTAL_CHROMOSOME - 1) else 0

    child = Genome()  # new child created from Genome class.
    # child.chromosome is initialized to instance variable.
    child.chromosome = child_chro
    # childs fitness value is return value from childs eval_chromosome value.
    child.fitness = eval_chromosome(child.chromosome)
    return child


# parameter is Genome type expect Genome object as return
def scramble_mutation(genome: Genome) -> Genome:
    subset_length = random.randint(2, 6)  # random int from 2 to 6
    # start point equal to 0 to TOTAL_CHROMOSOME - subset lenght
    start_point = random.randint(0, TOTAL_CHROMOSOME - subset_length)
    # subset_index list equal to start_point to start_point _ subset_length
    subset_index = [start_point, start_point + subset_length]
    # subset assigned to generated genome.chromosome[sliced by subset_index[0],to subset_index[1]]
    subset = genome.chromosome[subset_index[0]:subset_index[1]]

    random.shuffle(subset)  # return shuffled list

    # generated genomes chromosome list [sliced by subset_index[0]:subset_index[1]] assigned from subset
    genome.chromosome[subset_index[0]:subset_index[1]] = subset
    # genomes fitness value is return value from eval_chromosome
    genome.fitness = eval_chromosome(genome.chromosome)
    # return genome object.
    return genome


def reproduction(population: List[Genome]) -> Genome:
    parents = [tournament_selection(population, 30), random.choice(population)]
    # child genome return from crossover
    child = one_point_crossover(parents)

    # if random rage is smaller than mutation rate scramble_mutation apply.
    if random.random() < MUTATION_RATE:
        scramble_mutation(child)

    return child


def visualize(all_fittest: List[Genome], all_pop_size: List[int]):
    fig = plt.figure(tight_layout=True, figsize=(10, 6))
    gs = gridspec.GridSpec(2, 1)

    # Top grid: Route
    chromosome = [0] * (len(all_fittest[-1].chromosome) + 2)
    chromosome[1:-1] = all_fittest[-1].chromosome
    coordinates = [CITY_COORDINATES[i] for i in chromosome]
    x, y = zip(*coordinates)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(x, y, color="midnightblue")
    ax.scatter(x, y, color="midnightblue")

    for i, xy in enumerate(coordinates[:-1]):
        ax.annotate(i, xy, xytext=(-16, -4),
                    textcoords="offset points", color="tab:red")

    ax.set_title("Route")
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    # Bottom grid: Fitness & Populations
    ax = fig.add_subplot(gs[1, :])
    all_fitness = [genome.fitness for genome in all_fittest]
    ax.plot(all_fitness, color="midnightblue")

    color = 'tab:red'
    ax2 = ax.twinx()
    ax2.set_ylabel('Population size', color=color)
    ax2.plot(all_pop_size, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    at = AnchoredText(
        f"Best Fitness: {all_fittest[-1].fitness}", prop=dict(size=10),frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    ax.set_title("Fitness & Population Size")
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Generations")

    fig.align_labels()
    plt.grid(True)
    plt.savefig('berlin52_selection_parameter_50')
    plt.show()


if __name__ == "__main__":
    generation = 0

    population = [create_genome() for x in range(POPULATION_SIZE)]

    all_fittest = []
    all_pop_size = []

    while generation != MAX_GENERATION:
        generation += 1
        print(f"Generation: {generation} -- Population size: {len(population)} -- Best Fitness: {get_fittest_genome(population).fitness}")

        childs = []
        for _ in range(int(POPULATION_SIZE * 0.2)):
            child = reproduction(population)
            childs.append(child)
        population.extend(childs)

        # Kill weakness genome
        for genome in population:
            if genome.fitness > WEAKNESS_THRESHOLD:
                population.remove(genome)

        all_fittest.append(get_fittest_genome(population))
        all_pop_size.append(len(population))

    visualize(all_fittest, all_pop_size)
