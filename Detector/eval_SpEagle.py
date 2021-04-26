import sys
import os
sys.path.insert(0, os.path.abspath('../'))

from evaluators import *
from Detector.SpEagle import *
from yelpFeatureExtraction import *


"""
	SpEagle spam detection performance evaluation.
"""


def runSpEagle(new_priors, user_product_graph):
	"""
	Run SpEagle with the injected spams
	"""

	numerical_eps = 1e-5
	user_review_potential = np.log(np.array([[1 - numerical_eps, numerical_eps], [numerical_eps, 1 - numerical_eps]]))
	eps = 0.1
	review_product_potential = np.log(np.array([[1 - eps, eps], [eps, 1 - eps]]))

	potentials = {'u_r': user_review_potential, 'r_u': user_review_potential,
				  'r_p': review_product_potential, 'p_r': review_product_potential}

	model = SpEagle(user_product_graph, new_priors, potentials, message=None, max_iters=4)

	# new runbp func
	model.schedule(schedule_type='bfs')

	iter = 0
	while iter < 3:

		# set up additional number of iterations
		if iter == 0:
			num_bp_iters = 2
		else:
			num_bp_iters = 1

		message_diff = model.run_bp(start_iter=iter, max_iters=num_bp_iters)

		# print(message_diff)
		
		iter += num_bp_iters

		if message_diff < 1e-3:
			break

	return model
