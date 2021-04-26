import sys
import os
sys.path.insert(0, os.path.abspath('../'))

from Detector.gang import *

"""
	GANG spam detection performance evaluation.
"""

def runGANG(priors, user_product_graph, product_user_graph, user_ground_truth):
	"""
	Run GANG model
	"""

	# need normalized the prior before running GANG
	priors, mean_priors = nor_priors(priors)

	model = GANG(user_product_graph, product_user_graph, user_ground_truth,
				 priors, mean_priors, 0.1, nor_flg=True, sup_flg=False)

	# run Linearized Belief Propagation on product-user matrix with 1000 iterations
	model.pu_lbp(1000)

	return model, [priors[0], priors[1], priors[2]]
