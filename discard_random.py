import torch
class RandomDiscard():
    def __init__(self, device="cpu"):
        self.device = device

    def discard(self, hand):
        batch_size, draw_size, _ = hand.shape
        self.draw_probabilities = torch.ones((batch_size, draw_size))/draw_size
        random_indices_choice = torch.multinomial(self.draw_probabilities, 2, replacement=False)
        sampled_cards = self.select_cards_from_edges(hand, random_indices_choice)
        return sampled_cards
    
    def update(self, rewards):
        pass

    def select_cards_from_edges(self, hand, sampled_cards_indices):
        #should verify that this works
        batch_size, _ , _ = hand.shape
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
        cards = hand[batch_indices, sampled_cards_indices]
        return cards
    
    def train(self):
        pass

    def eval(self):
        pass
    
    def __call__(self, hand):
        return self.discard(hand)