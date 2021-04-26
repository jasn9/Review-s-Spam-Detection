import random as rd
import sys
import os
sys.path.insert(0, os.path.abspath('../'))


"""
	The implementation of the Random attack.
"""

def random_post_single_target(user_product_graph, c, r, target):
	"""
	user_product_graph: key = user_id, value = list of review tuples
	c: the list of the controlled accounts
	target: target businesses
	r: the number of reviews posted per target
	"""

	added_edges = []
	unique = 0
	account_log =[]

	# random adding edges
	
	selected_spammers = rd.sample(c ,r)
	for spammer in selected_spammers:
		user_product_graph[spammer].append((target, 1, -1, '2012-06-01'))
		added_edges.append((spammer, target))

	for account in selected_spammers:
		if account not in account_log:
			unique += 1

	account_log = account_log + selected_spammers

	return added_edges