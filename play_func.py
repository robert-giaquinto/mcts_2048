from __future__ import print_function, division
from ui import *
import numpy as np
from deap import base, creator, tools, algorithms
import multiprocessing
from collections import Sequence
from itertools import repeat
import random


def eval_mcts(params):
	args = {'rules': False, 'azmode': False, 'version': False, 'mode': None, 'resume': False}
	game = Game(**args)
	max_depth, max_iters = params[0], params[1]
	sco, tim, num = game.loop(strategy='mcts', delay=0, max_depth=max_depth, max_iters=max_iters)
	print(max_depth, ", ", max_iters, ", ", num, ", ", round(tim, 3), ", ", sco)
	return sco,


def two_params(icls, dl, dh, rl, rh):
	return icls([random.randint(dl, dh), random.randint(rl, rh)])


def mutGaussianInt(individual, mu, sigma, indpb):
	size = len(individual)
	if not isinstance(mu, Sequence):
		mu = repeat(mu, size)
	elif len(mu) < size:
		raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
	if not isinstance(sigma, Sequence):
		sigma = repeat(sigma, size)
	elif len(sigma) < size:
		raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))
	for i, m, s in zip(xrange(size), mu, sigma):
		if random.random() < indpb:
			new_ind = individual[i] + int(round(random.gauss(m, s)))
			if new_ind < 1:
				individual[i] = 1
			else:
				individual[i] = new_ind
	return individual,


def simple():
	pop = toolbox.population(n=pop_sz)
	hof = tools.HallOfFame(1000)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("min", np.min)
	stats.register("max", np.max)

	pop, logbook = algorithms.eaSimple(pop, toolbox,
		cxpb=cxpb, mutpb=mutpb, ngen=ngen,
		stats=stats, halloffame=hof, verbose=False)

	return pop, logbook, hof


# PARALLEL!
def parallel_simple(toolbox, pop_sz, cxpb, mutpb, ngen, n_jobs=4):
	# Process Pool of 4 workers
	print("Beginning parallel processing")
	pool = multiprocessing.Pool(processes=n_jobs)
	toolbox.register("map", pool.map)

	history = tools.History()
	# Decorate the variation operators
	toolbox.decorate("mate", history.decorator)
	toolbox.decorate("mutate", history.decorator)

	# Create the population and populate the history
	pop = toolbox.population(n=pop_sz)
	history.update(pop)
	# create a hall of fame record
	hof = tools.HallOfFame(1000)

	# intiialize stats
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	pop, logbook = algorithms.eaSimple(pop, toolbox,
		cxpb=cxpb, mutpb=mutpb, ngen=ngen,
		stats=stats, halloffame=hof, verbose=False)
	pool.close()
	return pop, logbook, hof, history
