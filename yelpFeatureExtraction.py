import math
from copy import deepcopy
from datetime import datetime
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

date_time_format_str = '%Y-%m-%d'

def MNR(data, data_type='user'):
	
	#	Normalized maximum number of reviews in a day for a user/product
	
	feature = {}
	for o_id, reviews in data.items():
		frequency = {}
		for review in reviews:
			if review[3] not in frequency:
				frequency[review[3]] = 1
			else:
				frequency[review[3]] += 1
		feature[o_id] = max(frequency.values())

	normalizer = max(feature.values())
	for k in feature.keys():
		feature[k] /= normalizer
	return feature

def PR_NR(data):
	feature = {}
	
	for o_id, reviews in data.items():
		positives = 0
		negatives = 0
		for review in reviews:
			if review[1] > 3:
				positives += 1
			else:
				if review[1] < 3:
					negatives += 1
		feature[o_id] = (float(positives)/len(reviews), float(negatives)/len(reviews))
	return feature

def avgRD_user(user_data, product_data):
	#	Average rating deviation of each user / product.
	p_avg = {}
	for o_id, reviews in product_data.items():
		p_avg[o_id] = np.mean(np.array([review[1] for review in reviews]))
	u_avgRD = {}
	for o_id, reviews in user_data.items():
		u_avgRD[o_id] = np.mean(np.array([abs(review[1] - p_avg[review[0]]) for review in reviews]))
	return u_avgRD

def avgRD_prod(product_data):
	#	Average rating deviation of each user / product.
	p_avg = {}
	for o_id, reviews in product_data.items():
		p_avg[o_id] = np.mean(np.array([review[1] for review in reviews]))

	p_avgRD = {}
	for o_id, reviews in product_data.items():
		p_avgRD[o_id] = np.mean(np.array([abs(review[1] - p_avg[o_id]) for review in reviews]))

	return p_avgRD

def BST(user_data):
	# Burstiness of reviews by users. Spammers are often short term
	bst = {}
	tau = 28.0  # 28 days
	for o_id, reviews in user_data.items():
		post_dates = sorted([datetime.strptime(review[3], date_time_format_str) for review in reviews])
		delta_days = (post_dates[-1] - post_dates[0]).days
		if delta_days > tau:
			bst[o_id] = 0.0
		else:
			bst[o_id] = 1.0 - (delta_days / tau)
	return bst

def ERD(data):
	#	Entropy of the rating distribution of each user (product)
	erd = {}
	for o_id, reviews in data.items():
		ratings = [review[1] for review in reviews]
		h, _ = np.histogram(ratings, bins=np.arange(1, 7))
		h = h / h.sum()
		h = h[np.nonzero(h)]
		erd[o_id] = (- h * np.log2(h)).sum()
	return erd

def ETG(data):
	#	Entropy of the gaps between any two consecutive ratings.
	etg = {}
	edges = [0, 0.5, 1, 4, 7, 13, 33]
	for o_id, reviews in data.items():
		if len(reviews) <= 1:
			etg[o_id] = 0
			continue
		posting_dates = sorted([datetime.strptime(review[3], date_time_format_str) for review in reviews])

		delta_days = [(posting_dates[i + 1] - posting_dates[i]).days for i in range(len(posting_dates) - 1)]
		delta_days = [d for d in delta_days if d < 33]

		h = []
		for delta in delta_days:
			j = 0
			while j < len(edges) and delta > edges[j]:
				j += 1
			h.append(j)
		_, h = np.unique(h, return_counts=True)
		if h.sum() == 0:
			etg[o_id] = 0
			continue
		h = h / h.sum()
		h = h[np.nonzero(h)]
		etg[o_id] = np.sum(- h * np.log2(h))
	return etg

def RD(product_data):
	# Calculate the deviation of the review ratings to the product average.
	rd = {}
	for o_id, reviews in product_data.items():
		avg = np.mean(np.array([review[1] for review in reviews]))
		for review in reviews:
			rd[(review[0], o_id)] = abs(review[1] - avg)
	return rd

def EXT(product_data):
	#	Whether a rating is extreme or not
	ext = {}
	for o_id, reviews in product_data.items():
		for review in reviews:
			if int(review[1]) == 5 or int(review[1]) == 1:
				ext[(review[0], o_id)] = 1
			else:
				ext[(review[0], o_id)] = 0
	return ext

def DEV(product_data):
	#	Deviation of each rating from the average rating of the target product.
	#	Need to use "recursive minimal entropy partitioning" to find beta_1

	beta_1 = 0.63
	dev = {}
	for o_id, reviews in product_data.items():
		p_avg_rating = np.mean(np.array([review[1] for review in reviews]))
		for review in reviews:
			u_id = review[0]  
			if (abs(p_avg_rating - review[1]) / 4.0 > 0.63):
				dev[(u_id, o_id)] = 1 
			else:
				dev[(u_id, o_id)] = 0
	return dev

def ETF(product_data):
	#	Binary feature: 0 if ETF_prime <= beta_3, 1 otherwise.
	#	ETF_prime = 1 - (date of last review of user i on product p from the date of the first review of the product / 7 months)
	
	beta_3 = 0.69

	first_time_product = {}
	for o_id, reviews in product_data.items():
		for review in reviews:
			if o_id not in first_time_product:
				first_time_product[o_id] = datetime.strptime(review[3], date_time_format_str)
			elif datetime.strptime(review[3], date_time_format_str) < first_time_product[o_id]:
				first_time_product[o_id] = datetime.strptime(review[3], date_time_format_str)

	etf = {} 
	for o_id, reviews in product_data.items():
		for review in reviews:
			td = datetime.strptime(review[3], date_time_format_str) - first_time_product[o_id]
			if (review[0], o_id) not in etf:
				etf[(review[0], o_id)] = td
			elif td > etf[(review[0], o_id)]:
				etf[(review[0], o_id)] = td

	for k, v in etf.items():
		if v.days > 7 * 30:
			etf[k] = 0
		elif 1.0 - v.days / (7 * 30) > beta_3:
			etf[k] = 1
		else:
			etf[k] = 0
	return etf

def ISR(user_data):
	#	Check if a user posts only one review
	isr = {}
	for o_id, reviews in user_data.items():
		# go through all review of this user
		for review in reviews:
			if len(reviews) == 1:
				isr[(o_id, review[0])] = 1
			else:
				isr[(o_id, review[0])] = 0
	return isr

def add_feature(existing_features, new_features, feature_names):
	#	Add or update feature(s) of a set of nodes of the same type to the existing feature(s).
	#	If a feature of a node is already is existing_features, then the new values will replace the existing ones.
	
	for k, v in new_features.items():
		if k not in existing_features:
			existing_features[k] = dict()
		for i in range(len(feature_names)):
			if len(feature_names) > 1:
				existing_features[k][feature_names[i]] = v[i]
			else:
				existing_features[k][feature_names[i]] = v

def construct_all_features(user_data, prod_data):
	#	Main entry to feature construction.
	
	UserFeatures = {}
	ProdFeatures = {}

	uf = MNR(user_data, data_type='user')
	add_feature(UserFeatures, uf, ["MNR"])
	pf = MNR(prod_data, data_type='prod')
	add_feature(ProdFeatures, pf, ["MNR"])

	uf = PR_NR(user_data)
	add_feature(UserFeatures, uf, ["PR", "NR"])
	pf = PR_NR(prod_data)
	add_feature(ProdFeatures, pf, ["PR", "NR"])

	uf = avgRD_user(user_data, prod_data)
	add_feature(UserFeatures, uf, ["avgRD"])
	pf = avgRD_prod(prod_data)
	add_feature(ProdFeatures, pf, ["avgRD"])
	
	uf = BST(user_data)
	add_feature(UserFeatures, uf, ["BST"])

	uf = ERD(user_data)
	add_feature(UserFeatures, uf, ["ERD"])
	pf = ERD(prod_data)
	add_feature(ProdFeatures, pf, ["ERD"])
	
	uf = ETG(user_data)
	add_feature(UserFeatures, uf, ["ETG"])
	pf = ETG(prod_data)
	add_feature(ProdFeatures, pf, ["ETG"])

	ReviewFeatures = {}
	rf = RD(prod_data)
	add_feature(ReviewFeatures, rf, ['RD'])

	rf = EXT(prod_data)
	add_feature(ReviewFeatures, rf, ['EXT'])

	rf = DEV(prod_data)
	add_feature(ReviewFeatures, rf, ['DEV'])

	rf = ETF(prod_data)
	add_feature(ReviewFeatures, rf, ['ETF'])

	rf = ISR(user_data)
	add_feature(ReviewFeatures, rf, ['ISR'])

	return UserFeatures, ProdFeatures, ReviewFeatures

def calculateNodePriors(feature_names, features_py, when_suspicious):
	#	Calculate priors of nodes P(y=1|node) using node features.
	
	priors = {}
	for node_id, v in features_py.items():
		priors[node_id] = 0

	for f_idx, fn in enumerate(feature_names):

		fv_py = []
		for node_id, v in features_py.items():
			if fn not in v:
				fv_py.append((node_id, -1))
			else:
				fv_py.append((node_id, v[fn]))
		fv_py = sorted(fv_py, key=lambda x: x[1])

		i = 0
		while i < len(fv_py):
			start = i
			end = i + 1
			while end < len(fv_py) and fv_py[start][1] == fv_py[end][1]:
				end += 1
			i = end

			for j in range(start, end):
				node_id = fv_py[j][0]
				if fv_py[j][0] == -1:
					priors[node_id] += pow(0.5, 2)
					continue
				if when_suspicious[fn] == '+1':
					priors[node_id] += pow((1.0 - float(start + 1) / len(fv_py)), 2)
				else:
					priors[node_id] += pow(float(end) / len(fv_py), 2)

	for node_id, v in features_py.items():
		priors[node_id] = 1.0 - math.sqrt(priors[node_id] / len(feature_names))
		if priors[node_id] > 0.999:
			priors[node_id] = 0.999
		elif priors[node_id] < 0.001:
			priors[node_id] = 0.001
	return priors
	