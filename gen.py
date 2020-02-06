import numpy as np

nb_sample_per_generation = 1000
nb_of_generation = 7000

percentage_population_to_keep = 0.15
random_mutation_percentage = 0.0015

a = "a_example.in"
b = "b_small.in"
c = "c_medium.in"
d = "d_quite_big.in"
e = "e_also_big.in"

nb_sample_to_keep_per_generation = int(nb_sample_per_generation * percentage_population_to_keep)
file_name = c


def load_pizzas(file_name):
    with open(file_name) as f:
        l = f.readline()
        [nb_max_slices, nb_max_pizza] = l.replace("\n", "").split(" ")
        l = f.readline()
        pizzas = np.array(l.replace("\n", "").split(" ")).astype(int)
    return int(nb_max_slices), int(nb_max_pizza), pizzas


def compute_scores(a_generation, pizzas):
    return a_generation.dot(pizzas)


def sort_generation(a_generation, nb_max_slices, max_nb_pizzas):
    def rank_a_popopulation(a_population, a_score):
        if a_population.sum() > max_nb_pizzas:
            return 0
        if a_score > nb_max_slices:
            return 0
        return a_score

    scores = compute_scores(a_generation, pizzas)
    ranks = [rank_a_popopulation(a_population, scores[i]) for i, a_population in enumerate(a_generation)]
    sorted_index = sorted(range(len(ranks)), key=lambda k: -ranks[k])
    return a_generation[sorted_index]


def mutate(sorted_populations):
    def merge(population_a, population_b):
        a_indexes = np.random.randint(2, size=population_a.shape[0])
        # copy is required otherwise it doesn't work (fuck references)
        return (population_a * a_indexes + (population_b * (1 - a_indexes))).copy()

    def random_mutation(generations):
        original_shape = generations.shape
        one_indexes = np.random.randint(generations.size, size=int(generations.size * random_mutation_percentage))
        zeros_indexes = np.random.randint(generations.size, size=int(generations.size * random_mutation_percentage))

        generations = generations.reshape(generations.size)
        generations[one_indexes] = 1
        generations[zeros_indexes] = 0
        generations = generations.reshape(original_shape)

        return generations

    new_generation = np.array(
        [merge(
            sorted_populations[int(np.random.rand() * nb_sample_to_keep_per_generation)],
            sorted_populations[int(np.random.rand() * nb_sample_to_keep_per_generation)]
        ) for i in range(nb_sample_per_generation)])
    new_generation = random_mutation(new_generation)
    return new_generation


def save_generation(a_generation, file_name):
    nb_slice = a_generation.sum()
    indexes = list(np.nonzero(a_generation)[0])
    indexes = " ".join(map(str, indexes))

    with open(f'out/{file_name}.out', 'w') as f:
        f.write(str(nb_slice) + '\n')
        f.write(indexes)


nb_max_slices, nb_max_pizza, pizzas = load_pizzas('in/' + file_name)

generation_shape = (nb_sample_per_generation, pizzas.shape[0])
generation_size = generation_shape[0] * generation_shape[1]
i = 0

generation = np.random.randint(2, size=generation_size).reshape(generation_shape)

best_population = generation[0].copy()
best_score = 0

print(f'max_slice: {nb_max_slices}, max_pizza: {nb_max_pizza}')
while i < nb_of_generation:
    i += 1

    generation = mutate(generation)
    generation = sort_generation(generation, nb_max_slices, nb_max_pizza)

    score = compute_scores(generation[0], pizzas)
    if score > best_score:
        best_score = score
        best_population = generation[0].copy()

        print(i, best_score, 1 - ((nb_max_slices - best_score) / nb_max_slices))
        save_generation(best_population, file_name)

        if best_score == nb_max_slices:
            break
