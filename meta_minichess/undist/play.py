import torch
from learning.muzero.mcts.mcts import Node, MCTS
from torch.nn import L1Loss
from scipy.stats import entropy
import numpy as np

def select_action(node, temperature=1, deterministic=True):
    visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
    action_probs = [visit_count_i ** (1 / temperature) for visit_count_i, _ in visit_counts]
    total_count = sum(action_probs)
    action_probs = [x / total_count for x in action_probs]
    if deterministic:
        action_pos = np.argmax([v for v, _ in visit_counts])
    else:
        action_pos = np.random.choice(len(visit_counts), p=action_probs)

    count_entropy = entropy(action_probs, base=2)
    return visit_counts[action_pos][1], count_entropy

class SharedStorage(object):
    def __init__(self, model):
        self.step_counter = 0
        self.model = model
        self.reward_log = []
        self.test_log = []
        self.eps_lengths = []
        self.temperature_log = []
        self.visit_entropies_log = []

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def state_dict(self):
        return self.model.state_dict()

    def incr_counter(self):
        self.step_counter += 1

    def get_counter(self):
        return self.step_counter

    def set_data_worker_logs(self, eps_len, eps_reward, temperature, visit_entropy):
        self.eps_lengths.append(eps_len)
        self.reward_log.append(eps_reward)
        self.temperature_log.append(temperature)
        self.visit_entropies_log.append(visit_entropy)

    def add_test_log(self, score):
        self.test_log.append(score)

    def get_worker_logs(self):
        if len(self.reward_log) > 0:
            reward = sum(self.reward_log) / len(self.reward_log)
            eps_lengths = sum(self.eps_lengths) / len(self.eps_lengths)
            temperature = sum(self.temperature_log) / len(self.temperature_log)
            visit_entropy = sum(self.visit_entropies_log) / len(self.visit_entropies_log)

            self.reward_log = []
            self.eps_lengths = []
            self.temperature_log = []
            self.visit_entropies_log = []

        else:
            reward = None
            eps_lengths = None
            temperature = None
            visit_entropy = None

        if len(self.test_log) > 0:
            test_score = sum(self.test_log) / len(self.test_log)
            self.test_log = []
        else:
            test_score = None

        return reward, eps_lengths, test_score, temperature, visit_entropy

class DataWorker(object):
    def __init__(self, rank, config, shared_storage, replay_buffer):
        self.rank = rank
        self.config = config
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer

    def run(self):
        model = self.config.get_uniform_network()
        with torch.no_grad():
            while self.shared_storage.get_counter() < self.config.training_steps:
                model.set_weights(self.shared_storage.get_weights())
                model.eval()
                env = self.config.new_game(self.config.seed + self.rank)

                obs = env.reset()
                done = False
                priorities = []
                eps_reward, eps_steps, visit_entropies = 0, 0, 0
                trained_steps = self.shared_storage.get_counter()
                _temperature = self.config.visit_softmax_temperature_fn(num_moves=len(env.history),
                                                                        trained_steps=trained_steps)
                while not done and eps_steps <= self.config.max_moves:
                    root = Node(0)
                    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    network_output = model.initial_inference(obs)
                    root.expand(env.to_play(), env.legal_actions(), network_output)
                    root.add_exploration_noise(dirichlet_alpha=self.config.root_dirichlet_alpha,
                                               exploration_fraction=self.config.root_exploration_fraction)
                    MCTS(self.config).run(root, env.action_history(), model)
                    action, visit_entropy = select_action(root, temperature=_temperature, deterministic=False)
                    obs, reward, done, info = env.step(action.index)
                    env.store_search_stats(root)

                    eps_reward += reward
                    eps_steps += 1
                    visit_entropies += visit_entropy

                    if not self.config.use_max_priority:
                        error = L1Loss(reduction='none')(network_output.value,
                                                         torch.tensor([[root.value()]])).item()
                        priorities.append(error + 1e-5)

                env.close()
                self.replay_buffer.save_game.remote(env,
                                                    priorities=None if self.config.use_max_priority else priorities)
                # Todo: refactor with env attributes to reduce variables
                visit_entropies /= eps_steps
                self.shared_storage.set_data_worker_logs.remote(eps_steps, eps_reward, _temperature, visit_entropies)