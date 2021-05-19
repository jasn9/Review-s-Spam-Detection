import gzip
import numpy as np
import random as rd
import copy as cp


import Attack.IncDS as incDS
import Attack.IncPR as incPR
import Attack.Random as randomA
import Attack.IncBP as incBP

from Detector.eval_Fraudar import *
from Detector.eval_GANG import *
from Detector.eval_SpEagle import *
from Detector.eval_fBox import *

from evaluators import *
from scipy.special import expit

def read_lines(path, convert_line):
    buf = []
    with gzip.open(path, 'rt') as file:
        for line in file:
            buf.append(convert_line(line))
    
    print("No of Lines read#", len(buf))
    return buf

def convert_line_to_review(line):
    review = line.strip().split()
    return review

def load_reviews(path):
    reviews = read_lines(path, convert_line_to_review)
    return reviews

def e_greedy_sample(attack_p, e=0.2):
    z = np.random.random()
    if z > e:
        attack = np.random.choice(list(attack_p), 1, p=list(attack_p.values()))
        attack = attack[0]
    else:
        attack = rd.choice(list(attack_p))
    return attack

def convert_review_to_user_graph(reviews):
    user_graph = {}
    for _, row in reviews.iterrows():
        if row['UserId'] not in user_graph:
            user_graph[row['UserId']] = []
        user_graph[row['UserId']].append((row['ProductId'], float(row['Rating']), row['Label'], row['Date']))
    return user_graph

def convert_review_to_product_graph(reviews):
    product_graph = {}
    for _, row in reviews.iterrows():
        if row['ProductId'] not in product_graph:
            product_graph[row['ProductId']] = []
        product_graph[row['ProductId']].append((row['UserId'], float(row['Rating']), row['Label'], row['Date']))
    return product_graph

def singleton_attack(user_product_graph, review_per_target, offset, target):
    added_edges = []
    new_id = 0
    for user in user_product_graph.keys():
        new_id = max(int(user), new_id)+1
    for i in range(0,review_per_target):
        added_edges.append((str(new_id + review_per_target * offset + i), target))
    return added_edges

def get_new_reviews(user_graph, product_graph, attack, elite_accounts, review_per_target, offset, target):
    if attack=='IncDS':
        new_reviews = incDS.ds_evasion_singe_target(user_graph, product_graph, elite_accounts, review_per_target, target)
    elif attack=='IncPR':
        new_reviews = incPR.pr_evasion_single_target(user_graph, product_graph, elite_accounts, review_per_target, target)
    elif attack=='Random':
        new_reviews = randomA.random_post_single_target(user_graph, list(elite_accounts), review_per_target, target)
    elif attack=='Singleton':
        new_reviews = singleton_attack(user_graph, review_per_target, offset, target)
    elif attack=='IncBP':
        new_reviews = incBP.bp_evasion_single_target(user_graph, product_graph, elite_accounts, review_per_target, target)
    return new_reviews

def add_new_reviews(user_graph, product_graph, new_reviews):
    evasive_spams = {}
    for new_review in new_reviews:
        user_id = new_review[0]
        product_id = new_review[1]
        label = -1
        rating = 5 # All are for promotions
        if user_id in user_graph:
            date = user_graph[user_id][0][3]
        else:
            date = '2012-06-01'
        if user_id not in user_graph:
            user_graph[user_id] = []
        user_graph[user_id].append((product_id, rating, label, date))
        product_graph[product_id].append((user_id, rating, label, date))
        if product_id not in evasive_spams:
            evasive_spams[product_id] = []
        evasive_spams[product_id].append((user_id, rating, label, date))
    
    return user_graph, product_graph, evasive_spams

def calulate_priors(user_graph, product_graph):
    UserFeatures, ProdFeatures, ReviewFeatures = construct_all_features(user_graph,product_graph)
    
    feature_config = {'MNR': '+1', 'PR':'+1', 'NR':'+1', 'avgRD':'+1', 'ERD':'-1', 'ETG':'-1', \
		'BST':'+1', 'RD':'+1', 'EXT':'+1', 'DEV':'+1','ETF':'+1', 'ISR': '+1'}
    user_priors = calculateNodePriors(['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG'], UserFeatures, feature_config)
    prod_priors = calculateNodePriors(['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG'], ProdFeatures, feature_config)
    review_priors = calculateNodePriors(['RD', 'EXT', 'EXT', 'DEV', 'ETF', 'ISR'], ReviewFeatures, feature_config)
    priors = [user_priors, review_priors, prod_priors]
    return priors

def run_detectors(user_graph, product_graph, new_reviews, detect_q, top_k_threshold):
    user_graph = cp.deepcopy(user_graph)
    product_graph = cp.deepcopy(product_graph)
    user_graph, product_graph, evasive_spams = add_new_reviews(user_graph, product_graph, new_reviews)
    user_ground_truth = create_evasion_ground_truth(user_graph, evasive_spams)
    
    priors = calulate_priors(user_graph, product_graph)

    detector_review_prob_map = {}
    for detector in detect_q:
        if detector == 'GANG':       
            print('Run GANG ...')
            gang_model, _ = runGANG(priors, user_graph, product_graph, user_ground_truth)
            gang_user_prob, _, gang_review_prob = gang_model.classify()
            detector_review_prob_map[detector] = scale(gang_review_prob)
        elif detector == 'Fraudar':
            print('Run Fraudar ...')
            fraudar_user_prob, fraudar_review_prob = runFraudar(priors, user_graph)
            detector_review_prob_map[detector] = scale(fraudar_review_prob)
        elif detector == 'SpEagle':
            print('Run SpEagle ...')
            speagle_model = runSpEagle(priors, user_graph)
            speagle_user_prob, speagle_review_prob, _ = speagle_model.classify()
            detector_review_prob_map[detector] = scale(speagle_review_prob)
        elif detector == 'fBox':
            print('Run fBox ...')
            fbox_user_prob, fbox_review_prob = runfBox(priors, user_graph)
            detector_review_prob_map[detector] = scale(fbox_review_prob)
        else:
            print('Run Prior ...')
            prior_user_prob, prior_review_prob = priors[0], priors[1]
            detector_review_prob_map[detector] = scale(prior_review_prob)

    sum_review_probs = {}
    spam_review_probs = {}
    detector_review_probs = {}
    reviews_iter = detector_review_prob_map[list(detector_review_prob_map.keys())[0]]
    for review in reviews_iter:
        sum_review_prob = 0.0
        detector_review_probs[review] = {}
        for detector in detector_review_prob_map:
            sum_review_prob += detect_q[detector] * detector_review_prob_map[detector][review]
            detector_review_probs[review][detector] = detector_review_prob_map[detector][review]
        
        sum_review_probs[review] = sum_review_prob
        spam_review_probs[review] = expit(sum_review_prob)


    sort_spam_probs = [(review, spam_review_probs[review]) for review in spam_review_probs]
    sort_spam_probs = sorted(sort_spam_probs, reverse=True, key=lambda x: x[1])

    top_k = int(len(sort_spam_probs)*top_k_threshold)
    
    top_k_reviews = [review[0] for review in sort_spam_probs[:top_k]]
    #print(new_reviews)
    for review in top_k_reviews:
        if review not in new_reviews:
            top_k_reviews.remove(review)
        else:
            print(review)

    #print('top k is ', len(top_k_reviews))
    
    return spam_review_probs, sum_review_probs, detector_review_probs, top_k_reviews, product_graph 

def remove_edges(graph, edges):
    new_graph = {}
    for key in graph:
        new_graph[key] = []
        for value in graph[key]:
            if (value[0], key) not in edges:
                new_graph[key].append(value)
    return new_graph

    
        


