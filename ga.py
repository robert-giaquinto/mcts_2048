from __future__ import print_function, division
from deap import base, creator, tools, algorithms
from play_func import eval_mcts, mutGaussianInt, parallel_simple, two_params
import numpy as np
import time

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# initialize individual parameters
d_low = 1
d_high = 25
r_low = 25
r_high = 150

# initialize mutation parameters
# i_low = 25
# i_high = 50
mu = 0
sigma = 5

# initialize search parameters
cxpb = 0.75
mutpb = 0.25
ngen = 25
pop_sz = 8

toolbox = base.Toolbox()
# toolbox.register("attr_param", random.randint, i_low, i_high)
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_param, n=2)
toolbox.register("individual", two_params, creator.Individual, dl=d_low, dh=d_high, rl=r_low, rh=r_high)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_mcts)
toolbox.register("mate", tools.cxOnePoint)
# toolbox.register("mutate", tools.mutUniformInt, low=i_low, up=i_high, indpb=0.25)
toolbox.register("mutate", mutGaussianInt, mu=mu, sigma=sigma, indpb=0.25)
toolbox.register("select", tools.selBest)


if __name__ == "__main__":
	start_time = time.time()
	pop, log, hof, history = parallel_simple(toolbox, pop_sz, cxpb, mutpb, ngen, n_jobs=4)
	print("total runtime:", time.time() - start_time)
	print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness.values[0]))
	print("Final population:\n", pop)
	print("Log:\n", log)

	output = []
	for i in hof:
		output.append(i + [i.fitness.values[0]])
		print(i + [i.fitness.values[0]])
	output = np.array(output)
	np.savetxt('ga_results_backup2.csv', output, fmt='%10.5f', delimiter=',')

	gene_order = []
	for i in history.genealogy_history:
		gene_order.append(history.genealogy_history[i])
	gene_order = np.array(gene_order)

	index = []
	for g in gene_order:
		for i, o in enumerate(output[:, 0:2]):
			if np.array_equal(g, o):
				index.append(i)
	genealogy_results = output[:, 2][index].tolist()

	if len(gene_order) == len(genealogy_results):
		output2 = []
		for i, g in enumerate(gene_order.tolist()):
			output2.append([i] + g + [genealogy_results[i]])
		output2 = np.array(output2)
		np.savetxt('ga_results2.csv', output2, fmt='%10.5f', delimiter=',')

	import matplotlib.pyplot as plt
	gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")
	plt.figure()
	plt.plot(gen, avg, label="average")
	plt.plot(gen, min_, label="minimum")
	plt.plot(gen, max_, label="maximum")
	plt.xlabel("Generation")
	plt.ylabel("Fitness")
	plt.legend(loc="lower right")
	plt.savefig('fitness.png', bbox_inches='tight')
