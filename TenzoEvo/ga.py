import models
import random
import pandas as pd
import ga_config


class GA:
    def __init__(self, model, measured_data, pop_size, num_of_populations, manipulator):
        self.__model: models.Model = model
        self.__measured_data = measured_data
        self.__populations = Populations()
        self.__pop_size = pop_size
        self.__num_of_populations = num_of_populations
        self.__number_of_peaks = measured_data['min'].notna().sum()
        self.__manipulator = manipulator

    def get_init_population(self):
        if self.__model.get_model_type() == 'dynamic_single_winkler':
            params = ga_config.DYNAMIC_SINGLE_WINKLER

        elif self.__model.get_model_type() == 'dynamic_double_pasternak':
            params = ga_config.DYNAMIC_DOUBLE_PASTERNAK

        pop = []
        for i in range(self.__pop_size):
            chromosome = []
            for param in params:
                if param[2] == 0:
                    gene = param[1]
                else:
                    gene = random.randrange(param[1], param[2])
                chromosome.append(gene)

            head = list(param[0] for param in params)

            for i in range(self.__number_of_peaks):
                q = ga_config.Q
                gene = random.randrange(q[1], q[2])
                chromosome.append(gene)
                head.append(f'{q[0]}{i + 1}')

            pop.append(chromosome)

        population = pd.DataFrame(pop, columns=head)
        return population

    def run_optimization(self):
        population = self.get_init_population()
        self.__optimize(population, 1, False)

    def __optimize(self, population, iteration, mutation):
        if iteration <= self.__num_of_populations:
            cols = list(population.columns)
            new_population = Population(cols, iteration, self.__model)
            for index, params in population.iterrows():
                print(f'population {iteration}, chromosome {index}')
                analytical_data = self.__model.calculate_model(params)
                # man = dtm.Manipulator(self.__measured_data.copy())

                # data manipulation
                self.__manipulator.set_analytical_data(analytical_data)
                # man.get_significant_points()
                # man.adjust_q_vector()

                # q_vector = man.get_adjusted_q_vector() * params['Q']

                self.__manipulator.move_and_superpose()
                super_data = self.__manipulator.get_superposed_resampled()
                measured_data = self.__manipulator.get_measured_data()

                # fitness calculation
                fitness = self.__calculate_fitness(measured_data, super_data)
                params['fitness'] = fitness
                # params = params.append(q_vector.iloc[0])
                new_population.add_chromosome(params)

            self.__populations.add_population(new_population)
            new_population.write_log()
            new_population.perform_selection()
            new_population.perform_crossover()
            iteration += 1

            if mutation:
                new_population.perform_mutation()
                return self.__optimize(new_population.get_mutation_product(), iteration, mutation)

            return self.__optimize(new_population.get_crossover_product(), iteration, mutation)

        best_params = self.__populations.get_last_population().get_max_fitness_row()
        best_analytical_data = self.__model.calculate_model(best_params)
        best_params.astype(int, errors='ignore').to_csv('logs/best_solution.txt')

        # man = self.__manipulator.Manipulator(self.__measured_data.copy())

        # data manipulation
        self.__manipulator.set_analytical_data(best_analytical_data)
        # man.get_significant_points()
        # man.adjust_q_vector()

        self.__manipulator.move_and_superpose()
        super_data = self.__manipulator.get_superposed_resampled()
        measured_data = self.__manipulator.get_measured_data()

        # plotter.plot_deflection_for_ga(measured_data, super_data)

        return None







class Population:
    def __init__(self, columns, population_id, model):
        self.__population_id = population_id
        self.__columns = columns
        self.__population = pd.DataFrame(columns=columns)
        self.__selection_product = None
        self.__crossover_product = None
        self.__mutation_product = None
        self.__model = model

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
