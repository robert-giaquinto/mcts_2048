from __future__ import print_function

import os
import os.path
import math
import time
import random
import copy
import strategy as ai
import numpy as np
# from joblib import Parallel, delayed


from colorama import init, Fore, Style
init(autoreset=True)

import keypress
from board import Board



class Game(object):
	"""
	A 2048 game
	"""

	__dirs = {
		keypress.UP:      Board.UP,
		keypress.DOWN:    Board.DOWN,
		keypress.LEFT:    Board.LEFT,
		keypress.RIGHT:   Board.RIGHT,
		keypress.SPACE:   Board.PAUSE,
	}

	__clear = 'cls' if os.name == 'nt' else 'clear'

	COLORS = {
		2:    Fore.GREEN,
		4:    Fore.BLUE + Style.BRIGHT,
		8:    Fore.CYAN,
		16:   Fore.RED,
		32:   Fore.MAGENTA,
		64:   Fore.CYAN,
		128:  Fore.BLUE + Style.BRIGHT,
		256:  Fore.MAGENTA,
		512:  Fore.GREEN,
		1024: Fore.RED,
		2048: Fore.YELLOW,
		# just in case people set an higher goal they still have colors
		4096: Fore.RED,
		8192: Fore.CYAN,
	}

	# see Game#adjustColors
	# these are color replacements for various modes
	__color_modes = {
		'dark': {
			Fore.BLUE: Fore.WHITE,
			Fore.BLUE + Style.BRIGHT: Fore.WHITE,
		},
		'light': {
			Fore.YELLOW: Fore.BLACK,
		},
	}

	SCORES_FILE = '%s/.term2048.scores' % os.path.expanduser('~')
	STORE_FILE = '%s/.term2048.store' % os.path.expanduser('~')

	def __init__(self, scores_file=SCORES_FILE, colors=COLORS,
				 store_file=STORE_FILE, clear_screen=True,
				 mode=None, azmode=False, **kws):
		"""
		Create a new game.
			scores_file: file to use for the best score (default
						 is ~/.term2048.scores)
			colors: dictionary with colors to use for each tile
			store_file: file that stores game session's snapshot
			mode: color mode. This adjust a few colors and can be 'dark' or
				  'light'. See the adjustColors functions for more info.
			other options are passed to the underlying Board object.
		"""
		self.board = Board(**kws)
		self.score = 0
		self.scores_file = scores_file
		self.store_file = store_file
		self.clear_screen = clear_screen

		self.__colors = colors
		self.__azmode = azmode

		self.loadBestScore()
		self.adjustColors(mode)

	def adjustColors(self, mode='dark'):
		"""
		Change a few colors depending on the mode to use. The default mode
		doesn't assume anything and avoid using white & black colors. The dark
		mode use white and avoid dark blue while the light mode use black and
		avoid yellow, to give a few examples.
		"""
		rp = Game.__color_modes.get(mode, {})
		for k, color in self.__colors.items():
			self.__colors[k] = rp.get(color, color)

	def loadBestScore(self):
		"""
		load local best score from the default file
		"""
		try:
			with open(self.scores_file, 'r') as f:
				self.best_score = int(f.readline(), 10)
		except:
			self.best_score = 0
			return False
		return True

	def saveBestScore(self):
		"""
		save current best score in the default file
		"""
		if self.score > self.best_score:
			self.best_score = self.score
		try:
			with open(self.scores_file, 'w') as f:
				f.write(str(self.best_score))
		except:
			return False
		return True

	def incScore(self, pts):
		"""
		update the current score by adding it the specified number of points
		"""
		self.score += pts
		if self.score > self.best_score:
			self.best_score = self.score

	def readMove(self):
		"""
		read and return a move to pass to a board
		"""
		k = keypress.getKey()
		return Game.__dirs.get(k)

	def store(self):
		"""
		save the current game session's score and data for further use
		"""
		size = self.board.SIZE
		cells = []

		for i in range(size):
			for j in range(size):
				cells.append(str(self.board.getCell(j, i)))

		score_str = "%s\n%d" % (' '.join(cells), self.score)

		try:
			with open(self.store_file, 'w') as f:
				f.write(score_str)
		except:
			return False
		return True

	def restore(self):
		"""
		restore the saved game score and data
		"""

		size = self.board.SIZE

		try:
			with open(self.store_file, 'r') as f:
				lines = f.readlines()
				score_str = lines[0]
				self.score = int(lines[1])
		except:
			return False

		score_str_list = score_str.split(' ')
		count = 0

		for i in range(size):
			for j in range(size):
				value = score_str_list[count]
				self.board.setCell(j, i, int(value))
				count += 1

		return True

	def loop(self, strategy=None, delay=None, max_depth=None, max_iters=None):
		"""
		main game loop. returns the final score.
		"""
		pause_key = self.board.PAUSE
		margins = {'left': 4, 'top': 4, 'bottom': 4}

		start_time = time.time()
		number_of_moves = 0
		try:
			while True:
				# only print if there's a delay or human player
				if delay or strategy is None:
					if self.clear_screen:
						os.system(Game.__clear)
					else:
						print("\n")
					print(self.__str__(margins=margins))
				if self.board.won() or not self.board.canMove():
					break

				# select move based on strategy employed
				if strategy == 'random':
					m = ai.random_move(self.board)
					if delay:
						time.sleep(delay)
				elif strategy == 'priority':
					m = ai.priority_move(self.board)
					if delay:
						time.sleep(delay)
				elif strategy == 'mcts':
					mcts = MCTS(self.board, max_depth=max_depth, max_iters=max_iters)
					m = mcts.search()

					if delay:
						time.sleep(delay)
				else:
					m = self.readMove()
				number_of_moves += 1

				if (m == pause_key):
					self.saveBestScore()
					if self.store():
						print("Game successfully saved. "
							  "Resume it with `term2048 --resume`.")
						return self.score
					print("An error occurred while saving your game.")
					return

				self.incScore(self.board.move(m))

		except KeyboardInterrupt:
			self.saveBestScore()
			return

		self.saveBestScore()
		if delay or strategy is None:
			print('You won!' if self.board.won() else 'Game Over')
		total_time = time.time() - start_time
		return self.score, total_time, number_of_moves

	def getCellStr(self, x, y):  # TODO: refactor regarding issue #11
		"""
		return a string representation of the cell located at x,y.
		"""
		c = self.board.getCell(x, y)

		if c == 0:
			return '.' if self.__azmode else '  .'

		elif self.__azmode:
			az = {}
			for i in range(1, int(math.log(self.board.goal(), 2))):
				az[2 ** i] = chr(i + 96)

			if c not in az:
				return '?'
			s = az[c]
		elif c == 1024:
			s = ' 1k'
		elif c == 2048:
			s = ' 2k'
		else:
			s = '%3d' % c

		return self.__colors.get(c, Fore.RESET) + s + Style.RESET_ALL

	def boardToString(self, margins={}):
		"""
		return a string representation of the current board.
		"""
		b = self.board
		rg = range(b.size())
		left = ' '*margins.get('left', 0)
		s = '\n'.join(
			[left + ' '.join([self.getCellStr(x, y) for x in rg]) for y in rg])
		return s

	def __str__(self, margins={}):
		b = self.boardToString(margins=margins)
		top = '\n'*margins.get('top', 0)
		bottom = '\n'*margins.get('bottom', 0)
		scores = ' \tScore: %5d  Best: %5d\n' % (self.score, self.best_score)
		return top + b.replace('\n', scores, 1) + bottom



# def _search_parallel(iter, available_branches, board_sim, max_depth):
# 	"""
# 	PRIVATE Function run the search algorithm in parallel
# 	:return:
# 	"""
# 	# randomly select a branch to search down
# 	branch = random.choice(available_branches)
# 	game_sim = Game(board_sim)
# 	depth = 0
# 	while True:
# 		if board_sim.won() or not board_sim.canMove() or depth > max_depth:
# 			return game_sim.score, branch, iter
# 		# first move is down the selected branch
# 		if depth == 0:
# 			next_move = branch
# 		else:
# 			# otherwise play out randomly
# 			available_moves = board_sim.get_valid_moves()
# 			next_move = random.choice(available_moves)
# 		# keep track of score based on move selection
# 		game_sim.incScore(board_sim.move(next_move))
# 		depth += 1

class MCTS(object):
	"""
	run mcts to find best move
	"""
	def __init__(self, board, max_depth, max_iters):
		self.max_depth = max_depth
		self.max_iters = max_iters
		self.board = board
		# self.n_jobs = n_jobs
		# self.verbose = verbose

	def search(self):
		# loop for max_iters
		available_branches = self.board.get_valid_moves()
		branch_scores = [0] * 4
		branch_counts = [0] * 4

		# search
		# if self.n_jobs > 1:
		# 	# in parallel
		# 	par_search = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
		# 		delayed(_search_parallel)(i, available_branches, self.board, self.max_depth)
		# 		for i in range(self.max_iters))
		# 	scores, branch, iters = zip(*par_search)
		# 	for i in iters:
		# 		branch_scores[branch[i] - 1] += scores[i]
		# 		branch_counts[branch[i] - 1] += 1
		# 	time_end = time.time()
		# else:
		# in sequence
		for i in xrange(self.max_iters):
			# randomly select a branch to search down
			branch = random.choice(available_branches)
			board_sim = copy.deepcopy(self.board)
			game_sim = Game(board_sim)
			depth = 0
			while True:
				if board_sim.won() or not board_sim.canMove() or depth > self.max_depth:
					branch_scores[(branch - 1)] += game_sim.score
					branch_counts[(branch - 1)] += 1
					break
				# first move is down the selected branch
				if depth == 0:
					next_move = branch
				else:
					# otherwise play out randomly
					available_moves = board_sim.get_valid_moves()
					next_move = random.choice(available_moves)
				# keep track of score based on move selection
				game_sim.incScore(board_sim.move(next_move))
				depth += 1

		# select move corresponding to best branch score
		branch_counts = np.array(branch_counts)
		branch_counts = np.where(branch_counts == 0, 1.0, branch_counts) # avoid divide by zero
		branch_results = np.array(branch_scores) / branch_counts
		move = np.where(branch_results == np.max(branch_results))[0][0] + 1
		return move
