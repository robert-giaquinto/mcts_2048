from __future__ import print_function, division
from ui import *
import numpy as np
import time
from joblib import Parallel, delayed


def _parallel_runs(game, strategy, delay, max_depth, max_iters):
	sco, tim, num = game.loop(strategy=strategy, delay=delay, max_depth=max_depth, max_iters=max_iters)
	print("Evaluated depth", max_depth, "over", max_iters, "iterations,",
		"with", num, "moves over", round(tim, 1),
		"seconds (avg", round(1000*tim/num, 1), "ms per move), score =", sco)
	return sco, tim, num

def start_game():
	"""
	Start a new game. If ``debug`` is set to ``True``, the game object is
	returned and the game loop isn't fired.
	"""
	args = sys.argv
	if len(args) == 2:
		strategy = str(args[1])
		if strategy == 'mcts':
			print("MCTS Selected without max_depth or max_iters parameters. Defaulting to 10, 50")
			print("Doing only one run")
			runs = 1
			delay = 0
			max_depth = 10
			max_iters = 50
			n_jobs = 1
		else:
			runs = 1
			delay = 0
			max_depth = None
			max_iters = None
			n_jobs = 1
	elif len(args) == 3:
		strategy = str(args[1])
		runs = int(args[2])
		if strategy == 'mcts':
			print("MCTS Selected without max_depth or max_iters parameters. Defaulting to 10, 50")
			delay = 0
			max_depth = 10
			max_iters = 50
			n_jobs = 1
		else:
			delay = 0
			max_depth = None
			max_iters = None
			n_jobs = 1
	elif len(args) == 4:
		strategy = str(args[1])
		runs = int(args[2])
		delay = float(args[3])
		if strategy == 'mcts':
			print("MCTS Selected without max_depth or max_iters parameters. Defaulting to 10, 50")
			max_depth = 10
			max_iters = 50
			n_jobs = 1
		else:
			max_depth = None
			max_iters = None
			n_jobs = 1
	elif len(args) == 6:
		strategy = str(args[1])
		runs = int(args[2])
		delay = float(args[3])
		max_depth = int(args[4])
		max_iters = int(args[5])
		n_jobs = 1
	elif len(args) == 7:
		strategy = str(args[1])
		runs = int(args[2])
		delay = float(args[3])
		max_depth = int(args[4])
		max_iters = int(args[5])
		n_jobs = int(args[6])
	else:
		strategy = 'human'
		runs = 1
		delay = .01
		max_depth = None
		max_iters = None
		n_jobs = 1

	# set args to standard values:
	args = {'rules': False, 'azmode': False, 'version': False, 'mode': None, 'resume': False}

	if n_jobs == 1:
		scores = []
		times = []
		num_moves = []
		for t in xrange(runs):
			game = Game(**args)
			sco, tim, num = game.loop(strategy=strategy, delay=delay, max_depth=max_depth, max_iters=max_iters)
			scores.append(sco)
			times.append(tim)
			num_moves.append(num)

			print("Evaluated depth", max_depth, "over", max_iters, "iterations,",
				"with", num, "moves over", round(tim, 1),
				"seconds (avg", round(1000*tim/num, 1), "ms per move), score =", sco)
	else:
		# in parallel
		game = Game(**args)
		search_pack = Parallel(n_jobs=n_jobs, verbose=0)(
			delayed(_parallel_runs)(game=game, strategy=strategy,
				delay=delay, max_depth=max_depth, max_iters=max_iters)
			for i in xrange(runs))

		scores, times, num_moves = zip(*search_pack)

	scores = np.array(scores)
	times = np.array(times)
	num_moves = np.array(num_moves)
	output = np.array([scores, times, num_moves]).T

	filename = strategy + '_' + str(runs) + '_runs_' + str(max_depth) + '_depth_' + str(max_iters) + 'iters.csv'
	np.savetxt(filename, output, fmt='%10.5f', delimiter=",")
	print("\nUsing strategy:", strategy,
		"averaged over", runs, "runs.")
	print("Average score =", round(scores.mean(), 1),
		"\nStandard deviation =", round(scores.std(), 1),
		"\nMax = ", scores.max(),
		"\nMin = ", scores.min(),
		"\nAverage time per game = ", round(times.mean(), 1),
		"\nAverage number of moves = ", round(num_moves.mean(), 1),
		"\nAverage time per move = ", round(1000* times.sum() / num_moves.sum(), 1), "ms",
		"\n")

start_game()
