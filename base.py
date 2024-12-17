#cards should be array of suite, number
import torch
import numpy as np
from scoring import cribbage_scoring

def card_deck(batch_size, device="cpu"):
    cards = torch.arange(1,14, device=device)
    suits = torch.arange(1,5,device=device)
    grid1, grid2 = torch.meshgrid(cards, suits,indexing='ij')
    deck = torch.column_stack([grid1.ravel(), grid2.ravel()])
    batch_deck = torch.repeat_interleave(deck[None,:,:],batch_size,dim=0)
    return batch_deck

def shuffle_deck(deck,device="cpu"):
    batch_size, num_rows, _ = deck.shape
    row_permutations = torch.stack([torch.randperm(num_rows,device=device) for _ in range(batch_size)])
    batch_indices = torch.arange(batch_size,device=device).unsqueeze(1).expand(-1, num_rows)
    shuffled_deck = deck[batch_indices, row_permutations]
    return shuffled_deck

def batch_draw(batch_size, deck, n,device="cpu"):
    num_cards, _ = deck.shape
    batched_uniform = torch.ones((batch_size, num_cards),device=device)/num_cards
    indicies = torch.multinomial(batched_uniform, n, replacement=False)
    return deck[indicies]

def card_to_visual(card):
    if card[0] == 1:
        num_str = "A"
    elif card[0] == 11:
        num_str = "J"
    elif card[0] == 12:
        num_str = "Q"
    elif card[0] == 13:
        num_str = "K"
    else:
        num_str = f"{card[0]}"
    
    if card[1] == 1:
        suite_str = "H"
    elif card[1] == 2:
        suite_str = "D"
    elif card[1] == 3:
        suite_str = "C"
    elif card[1] == 4:
        suite_str = "S"
    return num_str + " of " + suite_str

def hand_to_visual(hand):
    return str([card_to_visual(card) for card in hand])

def starting_visual(hand, flipped):
    print("Hand: " + hand_to_visual(hand))
    print("Flipped " + card_to_visual(flipped))
    hand_batched = hand[None,:,:]
    flipped_batched = flipped[None, :]
    points, explanation = cribbage_scoring(hand_batched, flipped_batched)
    print("Total Points: " + str(points[0]))
    print("Points Breakdown: " + str(explanation))

def batch_remove_cards(batch_cards, remove_cards):
    #batch_cards - batch_size, num_cards, 2
    #remove_cards - batch_size, num_remove_cards, 2
    #assumes that we know that each of the remove cards is inside the batch_cards
    batch_size,num_cards,_ = batch_cards.shape
    num_remove_cards = remove_cards.shape[1]
    broadcast_equals = batch_cards[:,:,None,:] == remove_cards[:,None,:,:]
    batch_in_mask = broadcast_equals.all(-1).any(-1)
    new_batch_cards = batch_cards[batch_in_mask == False].reshape(batch_size, num_cards - num_remove_cards, 2)
    return new_batch_cards

deck = card_deck(100)
drawn = batch_draw(100, deck[0], 7)
new_batch_cards = batch_remove_cards(deck, drawn)
print(new_batch_cards.shape)
#hands, flipped = drawn[:,:-1,:], drawn[:,-1,:]
#starting_visual(hands[0], flipped[0])

#discard_model = DiscardTransformer(32, 4, 3, 100)
#edge_logits, edge_indices = discard_model.forward(hands)
#sampled_edges, sample_edge_indices = sample_edge_nodes(edge_logits, edge_indices, 3)
