from dataclasses import dataclass
from random import choices, choice, uniform


@dataclass
class Item:
    value: float
    weight: float


@dataclass
class ProblemParams:
    max_weight: float
    population_size: int
    crossover_propability: float
    mutation_propability: float
    stop_threshold: float
    items: list[Item]


def random_decision(decision_size: int) -> list[bool]:
    return [
        bool(choice((0, 1))) for _ in range(decision_size)
    ]


def initalize_population(decision_size: int, population_size: int) -> list[list[bool]]:
    return [random_decision(decision_size) for _ in range(population_size)]


def fitness_function(decision: list[bool], param: ProblemParams) -> float:
    result, weight = 0, 0
    for item, is_active in zip(param.items, decision):
        if not is_active:
            continue
        result += item.value
        weight += item.weight
    return result if weight <= param.max_weight else 0


def assess_population(population: list[list[bool]], params: ProblemParams) -> float:
    return sum([fitness_function(decision, params) for decision in population])


def selection(population: list[list[bool]], params: ProblemParams) -> list[list[bool]]:
    propabilites = [fitness_function(decision, params) for decision in population]
    summed_props = sum(propabilites)

    if summed_props == 0:
        return population

    for i in range(len(propabilites)):
        propabilites[i] /= summed_props

    return choices(population, weights=propabilites, k=params.population_size)


def unique_pair(population: list[list[bool]]) -> tuple[int, int]:
    select = list(range(len(population)))
    i = choice(select)
    select.remove(i)
    j = choice(select)
    return (i, j)


def crossover(population: list[list[bool]], params: ProblemParams) -> list[list[bool]]:
    result = [list(decision) for decision in population]
    for _ in range(len(population) // 2):
        if params.crossover_propability < uniform(0, 1):
            continue

        father, mother = unique_pair(population)
        mask = random_decision(len(population[0]))

        for i, mask_bool in enumerate(mask):
            if not mask_bool:
                continue
            result[father][i], result[mother][i] = result[mother][i], result[father][i]
    return result


def mutation(population: list[list[bool]], params: ProblemParams) -> list[list[bool]]:
    result = [list(decision) for decision in population]
    for decision in result:
        for i, _ in enumerate(decision):
            if params.mutation_propability < uniform(0, 1):
                continue
            decision[i] = not decision[i]
    return result


def stop_condition(new_pops: list[list[bool]], old_pops: list[list[bool]], params: ProblemParams) -> bool:
    return abs(
        assess_population(new_pops, params) - assess_population(old_pops, params)
    ) <= params.stop_threshold


def solve(params: ProblemParams) -> list[bool]:
    population = initalize_population(len(params.items), params.population_size)
    i = 0
    while True:
        i += 1
        new_pops = selection(population, params)
        new_pops = crossover(new_pops, params)
        if stop_condition(new_pops, population, params):
            break
        population = new_pops
    print(f"Iterations: {i}")
    return max(population, key=lambda x: fitness_function(x, params))


def log_solution(solution: list[bool], params: ProblemParams):
    print(f"Solution = {solution}")
    items = [item for item, decision in zip(params.items, solution) if decision]
    print(f"Choosen items = {items}")
    print(f"Value = {sum([item.value for item in items])}")
    print(f"Weight = {sum([item.weight for item in items])}")


if __name__ == '__main__':
    params = ProblemParams(
        max_weight=30,
        population_size=100,
        crossover_propability=0.5,
        mutation_propability=0.05,
        stop_threshold=0.5,
        items=[
            Item(8, 5),
            Item(9, 7),
            Item(6, 7),
            Item(10, 3),
            Item(5, 2),
            Item(8, 9),
            Item(9, 8),
            Item(4, 12),
            Item(4, 17),
        ]
    )
    solution = solve(params)
    log_solution(solution, params)
