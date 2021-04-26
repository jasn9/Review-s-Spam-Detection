import time
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

from yelpFeatureExtraction import *

"""
	The implementation of the IncPR attack.
"""

def pr_evasion_single_target(user_product_graph, product_user_graph, c, r, target):
	"""
	Args:
		user_product_graph: key = user_id, value = list of review tuples
		product_user_graph: key = product_id, value = list of review tuples
		c: list of controlled accounts
		r: number of reviews to be posted each account
		t: target list
		feature_config: feature configuration file
	"""

	# total number of spams posted 
	added_edges = []
	t0 = time.time()

	# total number of selected unique accounts
	unique = 0
	# feature name list for account node
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']

	new_user_graph = {}
	new_product_graph = {}

	# compute the features and account priors
	UserFeatures, ProdFeatures, ReviewFeatures = construct_all_features(user_product_graph, product_user_graph)
	feature_config = {'MNR': '+1', 'PR':'+1', 'NR':'+1', 'avgRD':'+1', 'ERD':'-1', 'ETG':'-1', \
		'BST':'+1', 'RD':'+1', 'EXT':'+1', 'DEV':'+1','ETF':'+1', 'ISR': '+1'}
	new_upriors = calculateNodePriors(user_feature_list, UserFeatures, feature_config)

	# select r accounts with minimum priors
	selected_accounts = [(account, new_upriors[account]) for account in c]
	selected_accounts = sorted(selected_accounts, reverse = False, key=lambda x:x[1])
	selected_accounts = [account[0] for account in selected_accounts[:r]]

	# save the added edges
	for added_account in selected_accounts:
		review_id = (added_account, target)
		added_edges.append(review_id)
		
	t1 = time.time()
	
	return added_edges	