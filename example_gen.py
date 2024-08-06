"""
Author: Youssef Nassar
Date: 29 December 2023
"""
import matplotlib.pyplot as plt
import numpy as np
from hpob_handler import HPOBHandler
from methods.generative_hpo import GenerativeHPO
from tqdm import tqdm
import json

seeds = ["test0", "test1", "test2", "test3", "test4"]
acc_list = []
n_trials = 50

results_path = "results/"
method_name = "GEN-VAE"

hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3-test")
search_space_id = "5971" 
dataset_ids =  ["10093", "3954", "43", "34536", "9970", "6566"]
num_tokens = 17
dim_model = 32
n_batches = 100
batch_size = 16
acc_per_method = []
# defining nested dictionaries to  save results
search_space_id_dict = {}
results_dict = {}

for dataset_id in tqdm(dataset_ids):
    dataset_dict = {}
    for seed in tqdm(seeds):
        #define the HPO method
        method = GenerativeHPO(num_tokens, dim_model, n_batches, batch_size)

        #evaluate the HPO method
        acc = hpob_hdlr.evaluate_continuous(method, search_space_id = search_space_id, 
                                                dataset_id = dataset_id,
                                                seed = seed,
                                                n_trials = n_trials )
        acc_per_method.append(acc)
        dataset_dict[seed] = acc

    search_space_id_dict[dataset_id] = dataset_dict

    plt.plot(np.array(acc_per_method).mean(axis=0))
results_dict[search_space_id] = search_space_id_dict

#save resulted dictionary as json file
with open(results_path+method_name+".json", 'w') as json_file:
    json.dump(results_dict, json_file, indent=2)

plt.legend(dataset_ids)
plt.show()


