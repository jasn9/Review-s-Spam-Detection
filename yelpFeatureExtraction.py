import math
from copy import deepcopy
from datetime import datetime
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

date_time_format_str = '%Y-%m-%d'

def MNR(data, data_type='user'):
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
	p_avg = {}
	for o_id, reviews in product_data.items():
		p_avg[o_id] = np.mean(np.array([review[1] for review in reviews]))
	u_avgRD = {}
	for o_id, reviews in user_data.items():
		u_avgRD[o_id] = np.mean(np.array([abs(review[1] - p_avg[review[0]]) for review in reviews]))
	return u_avgRD

def avgRD_prod(product_data):
	p_avg = {}
	for o_id, reviews in product_data.items():
		p_avg[o_id] = np.mean(np.array([review[1] for review in reviews]))

	p_avgRD = {}
	for o_id, reviews in product_data.items():
		p_avgRD[o_id] = np.mean(np.array([abs(review[1] - p_avg[o_id]) for review in reviews]))

	return p_avgRD

def BST(user_data):
	bst = {}
	tau = 28.0  
	for o_id, reviews in user_data.items():
		post_dates = sorted([datetime.strptime(review[3], date_time_format_str) for review in reviews])
		delta_days = (post_dates[-1] - post_dates[0]).days
		if delta_days > tau:
			bst[o_id] = 0.0
		else:
			bst[o_id] = 1.0 - (delta_days / tau)
	return bst

def ERD(data):
	erd = {}
	for o_id, reviews in data.items():
		ratings = [review[1] for review in reviews]
		h, _ = np.histogram(ratings, bins=np.arange(1, 7))
		h = h / h.sum()
		h = h[np.nonzero(h)]
		erd[o_id] = (- h * np.log2(h)).sum()
	return erd

def ETG(data):
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
	rd = {}
	for o_id, reviews in product_data.items():
		avg = np.mean(np.array([review[1] for review in reviews]))
		for review in reviews:
			rd[(review[0], o_id)] = abs(review[1] - avg)
	return rd

def EXT(product_data):
	ext = {}
	for o_id, reviews in product_data.items():
		for review in reviews:
			if int(review[1]) == 5 or int(review[1]) == 1:
				ext[(review[0], o_id)] = 1
			else:
				ext[(review[0], o_id)] = 0
	return ext

def DEV(product_data):
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
	for k, v in new_features.items():
		if k not in existing_features:
			existing_features[k] = dict()
		for i in range(len(feature_names)):
			if len(feature_names) > 1:
				existing_features[k][feature_names[i]] = v[i]
			else:
				existing_features[k][feature_names[i]] = v

def construct_all_features(user_data, prod_data):
	UserFeatures = {}
	ProdFeatures = {}

	user_feature = MNR(user_data, data_type='user')
	add_feature(UserFeatures, user_feature, ["MNR"])
	product_feature = MNR(prod_data, data_type='prod')
	add_feature(ProdFeatures, product_feature, ["MNR"])

	user_feature = PR_NR(user_data)
	add_feature(UserFeatures, user_feature, ["PR", "NR"])
	product_feature = PR_NR(prod_data)
	add_feature(ProdFeatures, product_feature, ["PR", "NR"])

	user_feature = avgRD_user(user_data, prod_data)
	add_feature(UserFeatures, user_feature, ["avgRD"])
	product_feature = avgRD_prod(prod_data)
	add_feature(ProdFeatures, product_feature, ["avgRD"])
	
	user_feature = BST(user_data)
	add_feature(UserFeatures, user_feature, ["BST"])

	user_feature = ERD(user_data)
	add_feature(UserFeatures, user_feature, ["ERD"])
	product_feature = ERD(prod_data)
	add_feature(ProdFeatures, product_feature, ["ERD"])
	
	user_feature = ETG(user_data)
	add_feature(UserFeatures, user_feature, ["ETG"])
	product_feature = ETG(prod_data)
	add_feature(ProdFeatures, product_feature, ["ETG"])

	ReviewFeatures = {}
	review_feature = RD(prod_data)
	add_feature(ReviewFeatures, review_feature, ['RD'])

	review_feature = EXT(prod_data)
	add_feature(ReviewFeatures, review_feature, ['EXT'])

	review_feature = DEV(prod_data)
	add_feature(ReviewFeatures, review_feature, ['DEV'])

	review_feature = ETF(prod_data)
	add_feature(ReviewFeatures, review_feature, ['ETF'])

	review_feature = ISR(user_data)
	add_feature(ReviewFeatures, review_feature, ['ISR'])

	return UserFeatures, ProdFeatures, ReviewFeatures

def calculateNodePriors(feature_names, features_py, when_suspicious):
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
	