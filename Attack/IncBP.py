import time
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

from Detector.gang import *
from evaluators import *
from yelpFeatureExtraction import *


"""
	The implementation of the IncBP attack.
"""

def bp_evasion_single_target(user_product_graph, product_user_graph, controlled_accounts, num_reviews, target):
	"""
	:param user_product_graph: key = user_id, value = list of review tuples
	:param product_user_graph: key = product_id, value = list of review tuples
	:param controlled_accounts: a set of controlled elite accounts
	:param num_reviews: number of reviews needed for each target
	:param targets: target
	"""
	count = 0 # new edge counter
	added_edges = []
	account_log = []
	unique = 0 # selected unique accounts counter
	t0 = time.time()

	# feature and prior calculation
	review_feature_list = ['RD', 'EXT', 'EXT', 'DEV', 'ETF', 'ISR']
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
	product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']
	feature_config = {'MNR': '+1', 'PR':'+1', 'NR':'+1', 'avgRD':'+1', 'ERD':'-1', 'ETG':'-1', \
		'BST':'+1', 'RD':'+1', 'EXT':'+1', 'DEV':'+1','ETF':'+1', 'ISR': '+1'}
	
	UserFeatures, ProdFeatures, ReviewFeatures = construct_all_features(user_product_graph, product_user_graph)
	user_ground_truth = create_user_ground_truth(user_product_graph)

	# normalize the priors for GANG
	new_upriors = calculateNodePriors(user_feature_list, UserFeatures, feature_config)
	new_ppriors = calculateNodePriors(product_feature_list, ProdFeatures, feature_config)
	new_rpriors = calculateNodePriors(review_feature_list, ReviewFeatures, feature_config)
	priors = [new_upriors, new_rpriors, new_ppriors]
	priors, mean_priors = nor_priors(priors)

	# initiialize the GANG model
	global_gang = GANG(user_product_graph, product_user_graph, user_ground_truth,
					   priors, mean_priors, 0.1, nor_flg=True, sup_flg=False)
	
	# run Linearized Belief Propagation with GANG
	global_gang.pu_lbp(1000)

	# get node posterior
	global_posterior = global_gang.res_pu_spam_post_vector

	# time counter
	t1 = time.time()
	
	# select the accounts with minimum posteriors estimated by GANG
	selected_accounts = [(account, global_posterior[int(account)]) for account in controlled_accounts]
	selected_accounts = sorted(selected_accounts, reverse=False, key=lambda x: x[1])
	selected_accounts = [account[0] for account in selected_accounts[:num_reviews]]

	for added_account in selected_accounts:
		added_edges.append((added_account, target))

	return added_edges