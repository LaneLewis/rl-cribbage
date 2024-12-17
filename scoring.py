import torch
import torch.nn.functional as F
from itertools import combinations
from math import comb

def flush_points(batch_hands, batch_flipped):
    #could still speed this up with dropping indicies if a full suite is found
    #or find a way to get number of unique elements along axis in torch
    batch_size, hand_size,_ = batch_hands.shape
    def check_suite(suite):
        return torch.sum(batch_hands[:,:,1] == suite, dim=1) == hand_size
    for i in range(1,5):
        flush_found = check_suite(i)
        if i == 1:
            is_flush = flush_found
        else:
            is_flush = is_flush.bitwise_or(flush_found)
    #if flush found check if first card matched batch_flipped
    flipped_matches_flush = batch_hands[:,0,1] == batch_flipped[:,1]
    points = 4*is_flush.long()
    points += flipped_matches_flush.bitwise_and(is_flush).long()
    return points

def convert_to_summed_hot_number(batch_hands, batch_flipped, num_cards=13):
    hand_numbers = batch_hands[:,:,0]
    hot_numbers = F.one_hot(hand_numbers-1, num_classes=num_cards)
    hot_numbers_counts = torch.sum(hot_numbers, dim=1)
    batch_flipped_numbers = batch_flipped[:,0]
    hot_flipped = F.one_hot(batch_flipped_numbers-1, num_classes=num_cards)
    hot_numbers_counts = hot_numbers_counts + hot_flipped
    return hot_numbers_counts

def runs_of_size(hot_numbers_counts, run_size):
    #finds the number of runs of each size, removes all higher order runs
    batch_size,_ = hot_numbers_counts.shape
    unfolded = F.unfold(hot_numbers_counts[:,None,None,:].float(),kernel_size=(1,run_size),stride=1,padding=0)
    total_runs = unfolded.prod(dim=1)
    run_occured = (total_runs > 0).float()

    has_adjacent = F.conv1d(run_occured[:,None,:], torch.ones(1,1,3),padding=1).squeeze(1)>1
    mask = 1 - has_adjacent.float()
    filtered_runs = total_runs * mask

    has_higher_order_runs = torch.sum(has_adjacent, dim=1) > 0
    all_runs = torch.sum(filtered_runs, dim=1).long()
    return all_runs, has_higher_order_runs

def runs_points(hot_numbers_counts, hand_size=5):
    num_cards = hand_size + 1
    batch_size, _ = hot_numbers_counts.shape
    run_points = torch.zeros(batch_size)
    for run_size in range(3, num_cards+1):
        total_runs, _ = runs_of_size(hot_numbers_counts, run_size)
        run_points += total_runs*run_size
    return run_points

def fifteens_points(batch_hands, batch_flipped, hand_size = 5):
    batch_size = batch_hands.shape[0]
    num_cards = hand_size + 1
    numbers = torch.concat((batch_hands[:,:,0], batch_flipped[:,None,0]),dim=-1)
    #change all royalty into tens for counting
    numbers[(numbers == 11) | (numbers == 12) | (numbers == 13)] = 10
    fifteens_points = torch.zeros(batch_size)
    for k in range(2, hand_size+1):
        points = 2*fifteens_fixed_combination_size(numbers, num_cards, k)
        fifteens_points += points
    return fifteens_points
    
def fifteens_fixed_combination_size(numbers, num_cards, k=3):
    indices = list(combinations(range(num_cards), k))
    indices_tensor = torch.tensor(indices, dtype=torch.long)
    combs = numbers[:,indices_tensor]
    total = torch.sum(combs, dim = -1)
    return torch.sum((total == 15),dim=1)

def jack_point(batch_hands, batch_flipped):
    is_jack = batch_hands[:,:,0] == 11 
    matches_flipped_suit = batch_hands[:,:,1] == batch_flipped[:,1,None]
    is_jack_point = torch.bitwise_and(is_jack, matches_flipped_suit).long()
    return torch.sum(is_jack_point,dim=1)

def pairs_points(hot_number_counts):
    all_pairs = 0
    #max of 4 of a kind in a single deck
    for number_same in range(2, 4):
        pairs = torch.sum(hot_number_counts == number_same, dim=1)*comb(number_same,2)
        all_pairs += pairs
    return 2*all_pairs

def cribbage_scoring(batch_hands,batch_flipped):
    #make sure the hand size is ok or if we should include flips
    hand_size = 4
    f_points  = flush_points(batch_hands, batch_flipped)
    hot_number_counts = convert_to_summed_hot_number(batch_hands, batch_flipped, 13)
    run_points = runs_points(hot_number_counts, hand_size)
    ft_points = fifteens_points(batch_hands, batch_flipped,hand_size)
    j_point = jack_point(batch_hands, batch_flipped)
    p_points = pairs_points(hot_number_counts)
    total_points = f_points + run_points + ft_points + j_point + p_points
    explanation = {"runs":run_points, "fifteens":ft_points, "jack":j_point, "flush":f_points, "pairs":p_points}
    return total_points, explanation

