import torch
import torch.nn.functional as F
from itertools import combinations
from math import comb

class CribbageHandScorer():
    def __init__(self, hand_size, batch_size, device = "cpu"):
        self.hand_size = hand_size
        self.batch_size = batch_size
        self.device = device
        self.num_card_values = 13
        self.hand_size = 4
        self.combination_indices_tensors = {}
        #prebuilds indexing tensors for counting fifteens
        for k in range(2, self.hand_size+1):
            indices = list(combinations(range(self.hand_size + 1), k))
            indices_tensor = torch.tensor(indices, dtype=torch.long,device=self.device)
            self.combination_indices_tensors[k] = indices_tensor

    def flush_points(self, hands, flipped_cards):
        def check_suite(suite):
            return torch.sum(hands[:,:,1] == suite, dim=1) == self.hand_size
        for i in range(1,5):
            flush_found = check_suite(i)
            if i == 1:
                is_flush = flush_found
            else:
                is_flush = is_flush.bitwise_or(flush_found)
        #if flush found check if first card matched batch_flipped
        flipped_matches_flush = hands[:,0,1] == flipped_cards[:,1]
        points = 4*is_flush.long()
        points += flipped_matches_flush.bitwise_and(is_flush).long()
        return points
    
    def hot_number_counts(self, hands, flipped_cards):
        hand_numbers = hands[:,:,0]
        hot_numbers = F.one_hot(hand_numbers-1, num_classes=self.num_card_values)
        hot_numbers_totals = torch.sum(hot_numbers, dim=1)
        batch_flipped_numbers = flipped_cards[:,0]
        hot_flipped = F.one_hot(batch_flipped_numbers-1, num_classes=self.num_card_values)
        hot_numbers_totals = hot_numbers_totals + hot_flipped
        return hot_numbers_totals
    
    def runs_of_size(self, hot_counts, run_size):
        #finds the number of runs of each size, removes all higher order runs
        unfolded = F.unfold(hot_counts[:,None,None,:].float(),kernel_size=(1,run_size),stride=1,padding=0)
        total_runs = unfolded.prod(dim=1)
        run_occured = (total_runs > 0).float()

        has_adjacent = F.conv1d(run_occured[:,None,:], torch.ones(1,1,3,device=self.device),padding=1).squeeze(1)>1
        mask = 1 - has_adjacent.float()
        filtered_runs = total_runs * mask

        has_higher_order_runs = torch.sum(has_adjacent, dim=1) > 0
        all_runs = torch.sum(filtered_runs, dim=1).long()
        return all_runs, has_higher_order_runs
    
    def runs_points(self, hot_counts):
        num_cards = self.hand_size + 1
        run_points = torch.zeros(self.batch_size,device = self.device)
        for run_size in range(3, num_cards+1):
            total_runs, _ = self.runs_of_size(hot_counts, run_size)
            run_points += total_runs*run_size
        return run_points

    def fifteens_points(self, hands, flipped_cards):
        numbers = torch.concat((hands[:,:,0], flipped_cards[:,None,0]),dim=-1)
        #change all royalty into tens for counting
        numbers[(numbers == 11) | (numbers == 12) | (numbers == 13)] = 10
        fifteens_points = torch.zeros(self.batch_size, device= self.device)
        for k in range(2, self.hand_size+1):
            points = 2*self.fifteens_fixed_combination_size(numbers, k)
            fifteens_points += points
        return fifteens_points
    
    def fifteens_fixed_combination_size(self, numbers, k):
        indices_tensor = self.combination_indices_tensors[k]
        combs = numbers[:,indices_tensor]
        total = torch.sum(combs, dim = -1)
        return torch.sum((total == 15),dim=1)
    
    def jack_point(self, hands, flipped_cards):
        is_jack = hands[:,:,0] == 11 
        matches_flipped_suit = hands[:,:,1] == flipped_cards[:,1,None]
        is_jack_point = torch.bitwise_and(is_jack, matches_flipped_suit).long()
        return torch.sum(is_jack_point,dim=1)
    
    def pairs_points(self, hot_counts):
        all_pairs = 0
        #max of 4 of a kind in a single deck
        for number_same in range(2, 4):
            pairs = torch.sum(hot_counts == number_same, dim=1)*comb(number_same,2)
            all_pairs += pairs
        return 2*all_pairs
        
    def score(self, hands, flipped_cards):
        #note removed total points for just pairs
        hot_counts = self.hot_number_counts(hands, flipped_cards)
        f_points  = self.flush_points(hands, flipped_cards)
        run_points = self.runs_points(hot_counts)
        ft_points = self.fifteens_points(hands, flipped_cards)
        j_point = self.jack_point(hands, flipped_cards)
        p_points = self.pairs_points(hot_counts)
        total_points = f_points + run_points + ft_points + j_point + p_points
        explanation = {"runs":run_points, "fifteens":ft_points, "jack":j_point, "flush":f_points, "pairs":p_points}
        return run_points, explanation
