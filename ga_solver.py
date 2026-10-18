import random
import math
import matplotlib.pyplot as plt
from uav_env import create_environment

class UAVGeneticAlgorithm:
    def __init__(self, grid_size=50):
        self.grid_size = grid_size
        self.start_point = (0, 0)
        self.end_point = (grid_size - 1, grid_size - 1)
        self.grid = create_environment(grid_size)

    def create_chromosome(self, num_waypoints=5):
        chromosome = [self.start_point]
        for _ in range(num_waypoints):
            chromosome.append((random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)))
        chromosome.append(self.end_point)
        return chromosome

    def calculate_fitness(self, chromosome):
        total_dist, hazard_pen = 0, 0
        for i in range(len(chromosome)-1):
            p1, p2 = chromosome[i], chromosome[i+1]
            total_dist += math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
            if self.grid[p1[0]][p1[1]] == 1:
                hazard_pen += 500
        
        fitness = 10000 / (total_dist + hazard_pen + 1)
        return fitness

    def crossover(self, parent1, parent2):
        
        if len(parent1) > 3:
            cut = random.randint(1, len(parent1) - 2)
            child1 = parent1[:cut] + parent2[cut:]
            child2 = parent2[:cut] + parent1[cut:]
            return child1, child2
        return parent1, parent2

    def mutate(self, chromosome, mutation_rate=0.2):
       
        for i in range(1, len(chromosome) - 1): 
            if random.random() < mutation_rate:
                chromosome[i] = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        return chromosome

    def run_optimization(self, pop_size=50, generations=100):
        print(f"Initializing Population of {pop_size} drones...")
        population = [self.create_chromosome() for _ in range(pop_size)]
        best_fitness_history = []
        best_chromosome = None
        max_fitness = -1

        for gen in range(generations):
            fitness_scores = [self.calculate_fitness(chrom) for chrom in population]
            
            
            current_best_fit = max(fitness_scores)
            best_fitness_history.append(current_best_fit)
            if current_best_fit > max_fitness:
                max_fitness = current_best_fit
                best_chromosome = population[fitness_scores.index(current_best_fit)]

            
            new_population = []
            for _ in range(pop_size // 2):
                p1 = population[random.randint(0, pop_size-1)]
                p2 = population[random.randint(0, pop_size-1)]
                c1, c2 = self.crossover(p1, p2)
                new_population.extend([self.mutate(c1), self.mutate(c2)])
            population = new_population

        return best_chromosome, best_fitness_history

    def visualize_results(self, best_path, fitness_history):
      
        plt.figure(figsize=(14, 6))
        
       
        plt.subplot(1, 2, 1)
        plt.imshow(self.grid, cmap='Greys', interpolation='nearest')
        y_coords = [p[0] for p in best_path]
        x_coords = [p[1] for p in best_path] 
        plt.plot(x_coords, y_coords, marker='o', color='red', linewidth=2, label="Optimal Flight Path")
        plt.plot(x_coords[0], y_coords[0], marker='s', color='blue', markersize=10, label="Start")
        plt.plot(x_coords[-1], y_coords[-1], marker='*', color='green', markersize=12, label="Target")
        plt.title("Evolved UAV Flight Path (Avoiding Hazards)")
        plt.legend()

       
        plt.subplot(1, 2, 2)
        plt.plot(fitness_history, color='blue', linewidth=2)
        plt.title("Fitness Convergence Curve")
        plt.xlabel("Generations")
        plt.ylabel("Best Fitness Score")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ga = UAVGeneticAlgorithm()
    best_path, history = ga.run_optimization(pop_size=50, generations=100)
    print("\nOptimization Complete!")
    print("Best Path Coordinates:", best_path)
    print("Opening output visualizations...")
    ga.visualize_results(best_path, history)