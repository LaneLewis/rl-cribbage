from base import card_deck, shuffle_deck, batch_draw,batch_remove_cards
from discard_nn import DiscardTransformer, sample_edge_nodes
import torch.nn as nn
import torch
from scoring import cribbage_scoring
import matplotlib.pyplot as plt
from hand_score import CribbageHandScorer
from discard_random import RandomDiscard
import numpy as np
import torch.nn.functional as F
import os
#should plot occurances of pairs, fifteens etc compared to random over learning


class AdamInit():
    def __init__(self, lr, beta1, beta2):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
    
    def __call__(self, params):
        return torch.optim.Adam(params, lr=self.lr, betas = (self.beta1, self.beta2), maximize=True)

def linear_hyper_param_scheduler(start_value, end_value, epochs):
    def hyper_param(epoch):
        return max(start_value + (end_value - start_value)*epoch/epochs, end_value)
    return hyper_param

def bad_hyper_param_scheduler(start_value, end_value, epochs):
    def hyper_param(epoch):
        return start_value + (end_value - start_value)*epoch/epochs
    return hyper_param


def exponential_hyper_param_scheduler(start_value, decay_rate):
    torch_decay = torch.tensor(decay_rate)
    def hyper_param(epoch):
        return start_value*torch.exp(-torch_decay*epoch)
    return hyper_param

class DiscardPolicy():
    def __init__(self, discard_nn:DiscardTransformer, optimizer_init = AdamInit(0.1, 0.9, 0.999), hp_scheduler = linear_hyper_param_scheduler(0.0, 0.0, 1), device="cpu"):
        self.nn = discard_nn
        self.neg_ll_loss = nn.CrossEntropyLoss()
        self.discard_log_likelilhood = None
        self.optimizer = optimizer_init(self.nn.parameters())
        self.device = device
        self.nn.train()
        self.hyper_param_scheduler = hp_scheduler
        self.epoch = 0
        self.hyper_param = None

    def discard(self, hand):
        self.discard_edges_logits, discard_edges_indices  = self.nn(hand)
        sampled_edge, sampled_edge_indices  = sample_edge_nodes(self.discard_edges_logits, discard_edges_indices, 1, device = self.device)
        self.discard_log_likelihood = -self.neg_ll_loss(self.discard_edges_logits, sampled_edge.squeeze(1))
        sampled_cards_indices = sampled_edge_indices.squeeze(1)
        sampled_cards = self.select_cards_from_edges(hand, sampled_cards_indices)
        return sampled_cards
    
    def update(self, rewards):
        self.epoch += 1
        #make sure rewards are disconnected
        self.optimizer.zero_grad()
        J = torch.mean(rewards*self.discard_log_likelihood)
        self.hyper_param = self.hyper_param_scheduler(self.epoch)
        J = J + self.hyper_param*self.entropy()
        J.backward()
        self.optimizer.step()

    def select_cards_from_edges(self, hand, sampled_cards_indices):
        #should verify that this works
        batch_size, _ , _ = hand.shape
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
        cards = hand[batch_indices, sampled_cards_indices]
        return cards
    
    def train(self):
        self.nn.train()
    
    def eval(self):
        self.nn.eval()
    
    def entropy(self):
        probs = F.softmax(self.discard_edges_logits, dim=-1)
        return -torch.mean(torch.sum(probs*torch.log(probs),dim=-1))
    
    def __call__(self, hand):
        return self.discard(hand)

def normalize_rewards(rewards):
    return (rewards - torch.mean(rewards))/torch.std(rewards)
    
class CribbageGameStart():
    #all games are same length, should be able to batch very effectively
    def __init__(self, crib_policy:DiscardPolicy, no_crib_policy:DiscardPolicy, batch_size = 10000,device="cpu", train_crib_policy = True, train_no_crib_policy=True):
        self.crib_policy = crib_policy
        self.no_crib_policy = no_crib_policy
        self.scorer = CribbageHandScorer(4, batch_size, device)
        self.deck = card_deck(1, device=device)[0]
        self.batch_size = batch_size
        self.crib_player_train_rewards = []
        self.no_crib_player_train_rewards = []
        self.crib_train_rewards = []
        self.device = device
        self.train_crib_policy = train_crib_policy
        self.train_no_crib_policy = train_no_crib_policy


    def batch_step(self):
        drawn = batch_draw(self.batch_size, self.deck, 13, device=self.device)

        crib_player_cards = drawn[:,:6]
        no_crib_player_cards = drawn[:,6:-1]
        flipped_card = drawn[:,-1]

        crib_player_discard, no_crib_player_discard = self.agent_actions(crib_player_cards, no_crib_player_cards)
        updated_crib_player_cards, updated_no_crib_player_cards = self.update_environment(crib_player_cards, crib_player_discard, no_crib_player_cards, no_crib_player_discard)
        crib_player_rewards, no_crib_player_rewards, crib_rewards = self.rewards(updated_crib_player_cards, crib_player_discard, updated_no_crib_player_cards, no_crib_player_discard,flipped_card)

        self.crib_player_train_rewards.append(torch.mean(crib_player_rewards).clone().detach().cpu().numpy())
        self.crib_train_rewards.append(torch.mean(crib_rewards).clone().detach().cpu().numpy())
        self.no_crib_player_train_rewards.append(torch.mean(no_crib_player_rewards).clone().detach().cpu().numpy())

        self.update_agents(crib_player_rewards, no_crib_player_rewards, crib_rewards)

        print(f"Crib Player Mean Rewards: {self.crib_player_train_rewards[-1]}")
        print(f"No-Crib Player Mean Rewards: {self.no_crib_player_train_rewards[-1]}")
        print(f"Crib Mean Rewards: {self.crib_train_rewards[-1]}")


    def update_agents(self, crib_player_rewards, no_crib_player_rewards, crib_rewards):
        crib_player_rewards = crib_player_rewards# + crib_rewards
        no_crib_player_rewards = no_crib_player_rewards - crib_rewards
        if self.train_crib_policy:
            self.crib_policy.update(crib_player_rewards)
        if self.train_no_crib_policy:
            self.no_crib_policy.update(no_crib_player_rewards)
    
    def agent_actions(self, crib_player_cards,no_crib_player_cards):
        crib_player_discard = self.crib_policy(crib_player_cards)
        no_crib_player_discard = self.no_crib_policy(no_crib_player_cards)
        return crib_player_discard, no_crib_player_discard
    
    def update_environment(self,crib_player_cards, crib_player_discard, no_crib_player_cards, no_crib_player_discard):
        updated_crib_player_cards = batch_remove_cards(crib_player_cards, crib_player_discard)
        updated_no_crib_player_cards = batch_remove_cards(no_crib_player_cards, no_crib_player_discard)
        return updated_crib_player_cards, updated_no_crib_player_cards
    
    def rewards(self, updated_crib_player_cards, crib_player_discard, updated_no_crib_player_cards, no_crib_player_discard, flipped_card):
        crib = torch.concat((crib_player_discard, no_crib_player_discard),dim=1)
        crib_player_points, crib_player_explanation = self.scorer.score(updated_crib_player_cards, flipped_card)
        no_crib_player_points, no_crib_player_explanation = self.scorer.score(updated_no_crib_player_cards, flipped_card)
        crib_points, crib_explanation = self.scorer.score(crib,flipped_card)

        crib_player_rewards = crib_player_points.detach()
        no_crib_player_rewards = no_crib_player_points.detach()
        crib_rewards = crib_points.detach()
        return crib_player_rewards, no_crib_player_rewards, crib_rewards

device = "mps"
run_name = "testing_exponential_2"
run_dir = "./runs/" + run_name
os.makedirs(run_dir, exist_ok=True)

crib_player_policy = DiscardPolicy(DiscardTransformer(128, 8, 4, 256, 0.1, device=device),
                                   optimizer_init=AdamInit(1e-3, 0.9, 0.99),
                                   hp_scheduler=linear_hyper_param_scheduler(1.0, 0.0, 20),
                                   device=device)
no_crib_player_policy = RandomDiscard()#DiscardPolicy(DiscardTransformer(64, 4, 4, 128, 0.1, device=device),optimizer_init=AdamInit(1e-4, 0.9, 0.99), device=device)

game = CribbageGameStart(crib_player_policy, no_crib_player_policy,batch_size=100000, device=device, train_no_crib_policy=False)
entropy = []
for i in range(20000):
    game.batch_step()
    mean_e = game.crib_policy.entropy().detach().clone().cpu().numpy()
    entropy.append(mean_e)
    print(game.crib_policy.hyper_param)

    print(f"entropy: {mean_e}")
    plt.title("Average Points Over Training")
    plt.plot(game.crib_player_train_rewards,label="Average Crib Player Points")
    plt.plot(game.no_crib_player_train_rewards, label="Average No-Crib Points")
    plt.plot(game.crib_train_rewards, label="Average Crib Points")
    plt.legend()
    plt.savefig(f"{run_dir}/train_rewards.png")
    plt.cla()

    plt.title("Total Reward Over Training")
    plt.plot(np.array(game.crib_player_train_rewards) + np.array(game.crib_train_rewards), label="Total Crib Player Points")
    plt.legend()
    plt.savefig(f"{run_dir}/train_total_rewards.png")
    plt.cla()
    
    plt.title("Entropy Over Training")
    plt.plot(entropy)
    plt.savefig(f"{run_dir}/entropy.png")
    plt.cla()