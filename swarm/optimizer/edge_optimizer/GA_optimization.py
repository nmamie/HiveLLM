import torch
import torch.nn as nn
from tqdm import tqdm
import asyncio
import pickle
import time
import numpy as np
from swarm.optimizer.edge_optimizer.evo_tools.GA import GA


# Define fitness function
def fitness_function(particle):
    # Extract selected features == 1
    particle = np.clip(particle, 0, 1)
    selected_features_indices = np.nonzero(particle == 1)[0]
    
    # Train classifier using selected features
    if selected_features_indices.size == 0:
        return 0
    X_selected = X.iloc[:, selected_features_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    # rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf.fit(X_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return -accuracy  # Minimize 1-accuracy (i.e., maximize accuracy)

def utility_function(particle, swarm, evaluator, use_learned_order=False):
    evaluator.reset()
    tasks = []
    log_probs = []
    _graph, log_prob = swarm.connection_dist.realize(swarm.composite_graph, use_learned_order=use_learned_order)
    tasks.append(evaluator.evaluate(_graph, return_moving_average=True))
    log_probs.append(log_prob)
    results = loop.run_until_complete(asyncio.gather(*tasks))
    utilities.extend([result[0] for result in results])   
    
    
    # return negative utility for minimization
    return -utility


def optimize_ga(swarm, evaluator, population_size=100, num_iter=100, lr=1e-1, display_freq=10, batch_size=4, record=False, experiment_id='experiment', use_learned_order=False):
    # select edges based on particle probabilities
    pbar = tqdm(range(num_iter))
    utilities = []
    loop = asyncio.get_event_loop()       
        
    # run GA
    ga = GA(func=utility_function, n_dim=swarm.connection_dist.num_edges, lb=0, ub=1, max_iter=num_iter, size_pop=population_size, prob_mut=0.1, utilities=utilities, swarm=swarm, evaluator=evaluator, use_learned_order=use_learned_order)
    start_time = time.time()
    best_x, best_y = ga.run()
    print(time.time() - start_time)
    print('best_x:', best_x, '\n', 'best_y:', best_y)
    
    return best_x, best_y
    

def optimize_reinforce(swarm, evaluator, num_iter=100, lr=1e-1, display_freq=10, batch_size=4, record=False, experiment_id='experiment', use_learned_order=False):
    optimizer = torch.optim.Adam(swarm.connection_dist.parameters(), lr=lr)
    pbar = tqdm(range(num_iter))
    utilities = []
    loop = asyncio.get_event_loop()
    for step in pbar:
        evaluator.reset()
        optimizer.zero_grad()
        tasks = []
        log_probs = []
        for i in range(batch_size):
            _graph, log_prob = swarm.connection_dist.realize(swarm.composite_graph, use_learned_order=use_learned_order)
            tasks.append(evaluator.evaluate(_graph, return_moving_average=True))
            log_probs.append(log_prob)
        results = loop.run_until_complete(asyncio.gather(*tasks))
        utilities.extend([result[0] for result in results])
        if step == 0:
            moving_averages = np.array([np.mean(utilities) for _ in range(batch_size)])
        else:
            moving_averages = np.array([result[1] for result in results])
        loss = (-torch.stack(log_probs) * torch.tensor(np.array(utilities[-batch_size:]) - moving_averages)).mean()
        loss.backward()
        optimizer.step()

        if i % display_freq == display_freq - 1:
            print(f'avg. utility = {np.mean(utilities[-batch_size:]):.3f} with std {np.std(utilities[-batch_size:]):.3f}')
            if record:
                with open(f"result/crosswords/{experiment_id}_utilities_{step}.pkl", "wb") as file:
                    pickle.dump(utilities, file)
                torch.save(swarm.connection_dist.state_dict(), f"result/crosswords/{experiment_id}_edge_logits_{step}.pt")