
from scipy.special import expit
import numpy as np

def create_user_ground_truth(user_graph):
	user_ground_truth = {}
	for user_id in user_graph:
		user_ground_truth[user_id] = 0
		for review in user_graph[user_id]:
			if review[2] == -1: # label == -1
				user_ground_truth[user_id] = 1
			
	return user_ground_truth

def create_evasion_ground_truth(user_graph, evasive_spams):
    user_ground_truth = {}
    for user_id in user_graph:
        user_ground_truth[user_id] = 0
        for review in user_graph[user_id]:
            if review[2] == -1:
                user_ground_truth.pop(user_id)
                break
    
    for product in evasive_spams:
        for review in evasive_spams[product]:
            user_ground_truth[review[0]] = 1
    
    return user_ground_truth


def scale(values):
    ranked_val = [(key, values[key]) for key in values]
    ranked_val = sorted(ranked_val, key=lambda x:x[1])

    ranked_val_min, ranked_val_max = ranked_val[0][1], ranked_val[-1][1]
    for x in values:
        values[x] = (values[x]-ranked_val_min) / (ranked_val_max-ranked_val_min)
        if values[x] == 0:
            values[x] = 0 + 1e-7
        elif values[x] == 1:
            values[x] = 1-1e-7

    return values

def compute_revenue(product_graph, target_products, elite_accounts):
	average_product_rating = {}
	for product in product_graph:
		tot_rating = 0
		for review in product_graph[product]:
			tot_rating += int(review[1])
		if(len(product_graph[product])!=0):
			average_product_rating[product] = tot_rating/len(product_graph[product])
		else:
			average_product_rating[product] = 0
	
	average_rating = sum(average_product_rating.values())/len(average_product_rating)
	
	RI = {}
	ERI = {}
	for product in target_products:
		RI[product] = average_product_rating[product] - average_rating
		average_elite_rating = []
		for review in product_graph[product]:
			if review[0] in elite_accounts:
				average_elite_rating.append(review[1])
		if len(average_elite_rating) == 0:
			ERI[product] = 0
		else:
			ERI[product] = sum(average_elite_rating)/len(average_elite_rating)
	
	beta_0 = 0.035
	beta_1 = 0.036
	alpha = 0.09

	revenue = {}
	for product in target_products:
		revenue[product] = alpha + beta_0 * RI[product] + beta_1 * ERI[product]
	
	return RI, ERI, revenue

def compute_cost(old, new, new_reviews, elite_accounts):
	old_ri = old[0]
	old_eri = old[1]
	old_revenue = old[2]

	new_ri = new[0]
	new_eri = new[1]
	new_revenue = new[2]
	
	product_elite_count = {}
	product_review_count = {}
	for review in new_reviews:
		if review[0] in elite_accounts:
			if review[1] not in product_elite_count:
				product_elite_count[review[1]] = 0
			product_elite_count[review[1]] += 1
		if review[1] not in product_review_count:
			product_review_count[review[1]] = 0
		product_review_count[review[1]] += 1
	
	cost = {}
	beta_0 = 0.035
	beta_1 = 0.036
	for review in new_reviews:
		if new_revenue[review[1]] - old_revenue[review[1]] <= 0:
			cost[review] = 0
		elif review[0] in elite_accounts:
			cost[review] = beta_0 * ((new_ri[review[1]]-old_ri[review[1]])/product_review_count[review[1]])\
				+ beta_1 * ((new_eri[review[1]]-old_eri[review[1]])/product_elite_count[review[1]])
		else:
			cost[review] = beta_0 * ((new_ri[review[1]]-old_ri[review[1]])/product_review_count[review[1]])
	
	return cost
		
def compute_reward(old, new, products):
	
	old_revenue = old[2]
	new_revenue = new[2]

	PE = {}
	for product in products:
		pe = new_revenue[product] - old_revenue[product]
		PE[product] = max(pe, 0)
	
	avg_pe = sum(PE.values())/len(PE)
	z = max(PE.values())-0

	reward = {}
	for product in products:
		reward[product] = expit((PE[product]-avg_pe)/z)
	
	return reward

def update_p(attack_p, product_attack_map, rewards, learning_rate_attackors):
	
	counters = {}
	acc_rewards = {}
	for attack in attack_p:
		counters[attack] = 0
		acc_rewards[attack] = 0

	for product in rewards:
		attack = product_attack_map[product]
		counter = counters[attack]
		acc_reward = acc_rewards[attack]

		counter += 1
		acc_reward = acc_reward + 1/(counter) * (rewards[product]-acc_reward)	
		
		attack_p[attack] += learning_rate_attackors * acc_reward

		counters[attack] = counter
		acc_rewards[attack] = acc_reward
	
	for attack in attack_p:
		if attack_p[attack] < 0:
			attack_p[attack] = 0
	
	return attack_p

def update_q(detect_q, cost, detector_review_probs, sum_review_probs, new_reviews, learning_rate_detectors):
	for detect in detect_q:
		grad = 0
		for review in new_reviews:
			grad += -1*cost[review] * detector_review_probs[review][detect] * expit(-sum_review_probs[review])
		grad = grad/len(new_reviews)
		detect_q[detect] = detect_q[detect] - learning_rate_detectors*grad
	return detect_q

def normalize_map(dict):
	total = sum(dict.values())
	for key in dict:
		dict[key] = dict[key]/total
	return dict

def compute_pe(old, new, target_products):
	
	old_revenue = old[2]
	new_revenue = new[2]

	acc = 0
	for product in target_products:
		acc += new_revenue[product] - old_revenue[product]
	
	return acc

def compute_loss(cost, spam_review_probs):
	acc = 0
	for review in cost:
		acc += -1 * cost[review] * np.log(spam_review_probs[review])
	return acc/len(cost)

def nor_priors(priors):
	new_upriors, new_rpriors, new_ppriors = priors
	ranked_upriors = [(user, new_upriors[user]) for user in new_upriors.keys()]
	ranked_upriors = sorted(ranked_upriors, reverse=True, key=lambda x: x[1])
	ranked_rpriors = [(user, new_rpriors[user]) for user in new_rpriors.keys()]
	ranked_rpriors = sorted(ranked_rpriors, reverse=True, key=lambda x: x[1])
	ranked_ppriors = [(user, new_ppriors[user]) for user in new_ppriors.keys()]
	ranked_ppriors = sorted(ranked_ppriors, reverse=True, key=lambda x: x[1])
	u_max, u_mean, u_min = ranked_upriors[0][1], ranked_upriors[int(len(ranked_upriors) / 2)][1], ranked_upriors[-1][1]
	p_max, p_mean, p_min = ranked_ppriors[0][1], ranked_ppriors[int(len(ranked_ppriors) / 2)][1], ranked_ppriors[-1][1]
	r_max, r_mean, r_min = ranked_rpriors[0][1], ranked_rpriors[int(len(ranked_rpriors) / 2)][1], ranked_rpriors[-1][1]
	for i, p in priors[0].items():
		priors[0][i] = (p - u_min) / (u_max - u_min)
	for i, p in priors[1].items():
		priors[1][i] = (p - r_min) / (r_max - r_min)
	for i, p in priors[2].items():
		priors[2][i] = (p - p_min) / (p_max - p_min)

	return priors, [u_mean, r_mean, p_mean]