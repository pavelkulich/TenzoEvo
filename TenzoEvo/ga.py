import models
import random
import pandas as pd
import numpy as np
import ga_config
import matplotlib.pyplot as plt
from scipy import spatial
import itertools


class GA:
    def __init__(self, model, measured_data, pop_size, num_of_populations, manipulator):
        self.__model: models.Model = model
        self.__measured_data = measured_data
        self.__populations = Populations()
        self.__pop_size = pop_size
        self.__num_of_populations = num_of_populations
        self.__num_of_peaks = measured_data['min'].notna().sum()
        self.__peaks = measured_data[measured_data['min'].notna()].reset_index(drop=True)
        self.__manipulator = manipulator
        self.__best_solutions = []

    def get_init_population(self):
        if self.__model.get_model_type() == 'dynamic_single_winkler':
            params = ga_config.DYNAMIC_SINGLE_WINKLER

        elif self.__model.get_model_type() == 'dynamic_double_pasternak':
            params = ga_config.DYNAMIC_DOUBLE_PASTERNAK

        param_pop = []
        q_pop = []
        for i in range(self.__pop_size):
            param_chromosome = []
            q_chromosome = []
            for param in params:
                if param[2] == 0:
                    gene = param[1]
                else:
                    gene = random.randrange(param[1], param[2])
                param_chromosome.append(gene)

            param_pop.append(param_chromosome)
            param_head = list(param[0] for param in params)

            q_head = []
            for i in range(self.__num_of_peaks):
                q = ga_config.Q
                gene = random.randrange(q[1], q[2])
                q_chromosome.append(gene)
                q_head.append(f'{q[0]}{i + 1}')

            q_pop.append(q_chromosome)

        param_population = pd.DataFrame(param_pop, columns=param_head)
        q_population = pd.DataFrame(q_pop, columns=q_head)
        return param_population, q_population

    def run_optimization(self):
        param_population, q_population = self.get_init_population()
        population = Population(param_population, q_population)
        self.__optimize(population, 1, False)

    def __optimize(self, population, iteration, mutation):
        if iteration <= self.__num_of_populations:
            param_population = population.get_param_population()
            q_population = population.get_q_population()
            fitness_list = []

            for index, params in param_population.iterrows():
                print(f'iteration: {iteration}, round: {index + 1}')
                moved_analytical_data_list = []
                q_vector = q_population.iloc[index]
                counter = 0

                for _, q_value in q_vector.iteritems():
                    analytical_data = self.__model.calculate_model(params, q_value)
                    moved_analytical_data = self.__manipulator.move_analytical_data(analytical_data,
                                                                                    self.__peaks.iloc[counter])
                    moved_analytical_data_list.append(moved_analytical_data)
                    del moved_analytical_data
                    counter += 1

                del q_vector

                # data manipulation
                self.__manipulator.superpose_analytical_data(moved_analytical_data_list)
                super_data = self.__manipulator.get_superposed_resampled()
                measured_data = self.__manipulator.get_measured_data()

                del moved_analytical_data_list

                # plt.plot(measured_data)
                # plt.plot(super_data)
                # plt.show()

                # fitness calculation
                fitness = self.__calculate_fitness(measured_data, super_data)
                fitness_list.append(fitness)

            del q_population

            fitness_population = pd.DataFrame(fitness_list, columns=['fitness'])
            population.set_fitness_population(fitness_population)

            (pd.concat(
                [population.get_param_population(), population.get_q_population(), population.get_fitness_population()],
                axis=1)).to_csv(f'log/iteration_{iteration}', sep=';')
            pd.concat(population.get_best_solution(), axis=0).to_csv(f'log/bs_iteration_{iteration}', sep=';')
            self.__best_solutions.append(population.get_best_solution())
            self.plot_best_solution(iteration)

            if iteration < self.__num_of_populations:
                population.perform_selection()
                population.perform_crossover()

                new_population = Population(population.get_param_crossover_product(),
                                            population.get_q_crossover_product())
                # print(f"best round {iteration} solution", population.get_best_solution())
                del population

                iteration += 1
                return self.__optimize(new_population, iteration, mutation)

    # def __calculate_fitness(self, measured_data, analytical_data):
    #     result = 1 - spatial.distance.cosine(measured_data['y_axis'], analytical_data['y_axis'])
    #     return int(result * 10000)

    def __calculate_fitness(self, measured_data, analytical_data):
        diff = np.sum((measured_data['y_axis'] / 100000 - analytical_data['y_axis'] / 100000) ** 2)
        print(diff)
        lmbd = lambda x: 1 if x <= 0 else x
        return lmbd(int(diff ** (-1)))

    def plot_best_solution(self, iteration=1):
        best_solution = self.__best_solutions[-1]
        params = best_solution[0]
        q_vector = best_solution[1]
        moved_analytical_data_list = []
        counter = 0

        for _, q_value in q_vector.iteritems():
            analytical_data = self.__model.calculate_model(params, q_value)
            moved_analytical_data = self.__manipulator.move_analytical_data(analytical_data,
                                                                            self.__peaks.iloc[counter])
            moved_analytical_data_list.append(moved_analytical_data)
            del moved_analytical_data
            counter += 1

        self.__manipulator.superpose_analytical_data(moved_analytical_data_list)
        super_data = self.__manipulator.get_superposed_resampled()
        measured_data = self.__manipulator.get_measured_data()

        plt.plot(measured_data['x_axis'], measured_data['y_axis'])
        plt.plot(super_data['x_axis'], super_data['y_axis'])
        plt.grid(True)
        plt.savefig(f'plots/iteration_{iteration}.png')
        plt.close()
        # plt.show()




class Population:
    def __init__(self, param_population, q_population):
        self.__param_population = param_population
        self.__q_population = q_population
        self.__fitness_population = None
        self.__param_selection_product = None
        self.__q_selection_product = None
        self.__param_crossover_product = None
        self.__q_crossover_product = None
        self.__mutation_product = None

    def get_param_population(self):
        return self.__param_population

    def get_q_population(self):
        return self.__q_population

    def get_fitness_population(self):
        return self.__fitness_population

    def get_param_crossover_product(self):
        return self.__param_crossover_product

    def get_q_crossover_product(self):
        return self.__q_crossover_product

    def get_best_solution(self):
        if self.__fitness_population is not None:
            idx = self.__fitness_population.idxmax(axis=0).at['fitness']
            param_solution = self.__param_population.iloc[idx]
            q_solution = self.__q_population.iloc[idx]
            fitness_solution = self.__fitness_population.iloc[idx]
            return param_solution, q_solution, fitness_solution

    def set_fitness_population(self, fitness_population):
        self.__fitness_population = fitness_population

    def has_fitness_population(self):
        if self.__fitness_population:
            return True

    def perform_selection(self, exp_pop_growth=False):
        weight_cum = np.cumsum(self.__fitness_population['fitness'])
        weight_sum = np.sum(self.__fitness_population['fitness'])

        if exp_pop_growth:
            selected_offs = self.__param_population.shape[0]
        else:
            selected_offs = int(self.__param_population.shape[0] / 8)

        indexes = []
        param_offs = []
        q_offs = []
        fit = []

        for i in range(selected_offs):
            idx = 0
            rand = random.randrange(1, weight_sum + 1)
            for j in weight_cum:
                if j >= rand:
                    break
                idx += 1
            indexes.append(idx)
            param_offs.append(self.__param_population.loc[idx].values)
            q_offs.append(self.__q_population.loc[idx].values)
            fit.append(self.__fitness_population.loc[idx].values)
        self.__param_selection_product = pd.DataFrame(param_offs, columns=self.__param_population.columns)
        self.__q_selection_product = pd.DataFrame(q_offs, columns=self.__q_population.columns)

    def perform_crossover(self, param_cross_point='h', q_cross_point='h'):
        if param_cross_point == 'q':
            param_crossing_point = int((self.__param_selection_product.shape[1]) / 4)
        elif param_cross_point == 'tq':
            param_crossing_point = int((self.__param_selection_product.shape[1]) / 4 * 3)
        else:
            param_crossing_point = int((self.__param_selection_product.shape[1]) / 2)

        if q_cross_point == 'q':
            q_crossing_point = int((self.__q_selection_product.shape[1]) / 4)
        elif q_cross_point == 'tq':
            q_crossing_point = int((self.__q_selection_product.shape[1]) / 4 * 3)
        else:
            q_crossing_point = int((self.__q_selection_product.shape[1]) / 2)

        def cross(pop, crossing_point, idx):
            item = []
            if not idx % 2:
                item.extend(pop.iloc[idx, :crossing_point].values)
                item.extend(pop.iloc[idx + 1, crossing_point:].values)
            else:
                item.extend(pop.iloc[idx, :crossing_point].values)
                item.extend(pop.iloc[idx - 1, crossing_point:].values)
            # item.extend([-1])
            return item

        param_offs = []
        q_offs = []

        for row in range(0, self.__param_selection_product.shape[0], 2):
            param_chrom1 = cross(self.__param_selection_product, param_crossing_point, row)
            param_chrom2 = self.__param_selection_product.iloc[row].tolist()
            param_chrom3 = cross(self.__param_selection_product, param_crossing_point, row + 1)
            param_chrom4 = self.__param_selection_product.iloc[row + 1].tolist()

            q_chrom1 = self.__q_selection_product.iloc[row].tolist()
            q_chrom2 = cross(self.__q_selection_product, q_crossing_point, row)
            q_chrom3 = self.__q_selection_product.iloc[row + 1].tolist()
            q_chrom4 = cross(self.__q_selection_product, q_crossing_point, row + 1)

            param_help_offs = []
            q_help_offs = []
            param_help_offs.append(param_chrom1)
            param_help_offs.append(param_chrom2)
            param_help_offs.append(param_chrom3)
            param_help_offs.append(param_chrom4)
            q_help_offs.append(q_chrom1)
            q_help_offs.append(q_chrom2)
            q_help_offs.append(q_chrom3)
            q_help_offs.append(q_chrom4)

            help_list = []
            help_list.append(param_help_offs)
            help_list.append(q_help_offs)

            offs_list = list(itertools.product(*help_list))
            for chrom in offs_list:
                param_offs.append(chrom[0])
                q_offs.append(chrom[1])

        self.__param_crossover_product = pd.DataFrame(param_offs, columns=self.__param_selection_product.columns)
        self.__q_crossover_product = pd.DataFrame(q_offs, columns=self.__q_selection_product.columns)
        # print(self.__param_crossover_product)
        # print(self.__q_crossover_product)


class Population2:
    def __init__(self, param_population, q_population, fitness_population):
        self.__param_population = param_population
        self.__q_population = q_population
        self.__fitness_population = fitness_population
        self.__selection_product = None
        self.__crossover_product = None
        self.__mutation_product = None

    def add_chromosome(self, chromosome):
        self.__population = self.__population.append(chromosome, ignore_index=True)

    def perform_selection(self):
        weight_cum = np.cumsum(self.__population['fitness'])
        weight_sum = np.sum(self.__population['fitness'])
        print(weight_cum)
        print(weight_sum)

        indexes = []
        offs = []
        for i in range(self.__population.shape[0]):
            idx = 0
            rand = random.randrange(1, weight_sum + 1)
            for j in weight_cum:
                if j >= rand:
                    break
                idx += 1
            indexes.append(idx)
            offs.append(self.__population.loc[idx].values)
        self.__selection_product = pd.DataFrame(offs, columns=self.__population.columns)

    def perform_crossover(self):
        cross_pop = self.__selection_product[self.__columns]
        # print(cross_pop)
        crossing_point = int((cross_pop.shape[1] - 1) / 2)

        def cross(pop, c_p, idx):
            item = []
            if not idx % 2:
                item.extend(pop.iloc[idx, :c_p + 1].values)
                item.extend(pop.iloc[idx + 1, c_p + 1:].values)
            else:
                item.extend(pop.iloc[idx, :c_p + 1].values)
                item.extend(pop.iloc[idx - 1, c_p + 1:].values)
            # item.extend([-1])
            return item

        offs = []
        for row in range(0, cross_pop.shape[0], 2):
            chrom_1 = cross(cross_pop, crossing_point, row)
            chrom_2 = cross(cross_pop, crossing_point, row + 1)
            offs.append(chrom_1)
            offs.append(chrom_2)

        self.__crossover_product = pd.DataFrame(offs, columns=cross_pop.columns)

    def perform_mutation(self):
        mut_pop = self.__crossover_product[self.__columns]

        def mutate(chromosome):
            new_chromosome = self.get_randomized_populatuion(chromosome)
            return new_chromosome

        offs = []
        for row in range(0, mut_pop.shape[0]):
            if row % 2:
                mut_row = mutate(mut_pop.iloc[row])
                offs.append(mut_row)
                continue
            offs.append(mut_pop.iloc[row])

        self.__mutation_product = pd.DataFrame(offs, columns=mut_pop.columns)

    def get_randomized_populatuion(self, chromosome):
        if self.__model.get_model_type() == 'dynamic_single_winkler':
            params = ga_config.DYNAMIC_SINGLE_WINKLER

        elif self.__model.get_model_type() == 'dynamic_double_pasternak':
            params = ga_config.DYNAMIC_DOUBLE_PASTERNAK

        if params:
            param = random.choice(params)
            gene = random.randrange(param[1], param[2])
            return chromosome.replace(to_replace=chromosome[param[0]], value=gene)

        else:
            print('Cannot randomize population. Model not implemented')
            return chromosome

    def get_population(self):
        return self.__population[self.__columns]

    def get_selection_product(self):
        return self.__selection_product[self.__columns]

    def get_crossover_product(self):
        return self.__crossover_product[self.__columns]

    def get_mutation_product(self):
        return self.__mutation_product[self.__columns]

    def get_max_fitness_row(self):
        return self.__population.loc[self.__population['fitness'] == np.max(self.__population['fitness']), :].iloc[0]

    def write_log(self):
        int_population = self.__population.astype(int, errors='ignore')
        int_population.to_csv(f'logs/population_{self.__population_id}.txt')


class Populations:
    def __init__(self):
        self.__population_id = -1
        self.__populations = []

    def add_population(self, population):
        self.__population_id += 1
        self.__populations.append(population)
        # self.__populations = self.__populations.append({self.__population_id, population.get_population().values},
        #                                                ignore_index=True)

    def get_last_population(self):
        return self.__populations[self.__population_id]

    def get_populations(self):
        return self.__populations

    def get_last_population_id(self):
        return self.__population_id

    def save_report(self):
        self.__populations.to_csv('report.csv')
