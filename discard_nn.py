import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def sample_edge_nodes(edge_logits, edge_indices, samples, device="cpu"):
    edge_probabilities = F.softmax(edge_logits, dim=-1)
    sampled_edges = torch.multinomial(edge_probabilities, samples, replacement=True).to(device)
    sampled_edge_indices = edge_indices[sampled_edges]
    return sampled_edges, sampled_edge_indices

class EdgeMapping(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def forward(self, embeddings):
        node_indices = torch.arange(embeddings.shape[1], device=self.device)
        node_combinations = torch.combinations(node_indices, 2)
        #embeddings - batch_size, nodes, embedding_dim
        pairwise_combinations = embeddings[:,node_combinations,:]
        edge_logits = torch.sum(pairwise_combinations[:,:,0,:]*pairwise_combinations[:,:,1,:],dim=-1)
        return edge_logits, node_combinations

class DiscardTransformer(nn.Module):
    def __init__(self, embed_size=512, num_heads=8, num_layers=6, ff_hidden=2048, dropout=0.1,device="cpu"):
        super().__init__()
        self.num_card_values = 13
        self.num_suites = 4
        # Token embedding and positional encoding
        self.card_value_embedding = nn.Embedding(self.num_card_values, embed_size, device=device)
        self.card_suite_embedding = nn.Embedding(self.num_suites, embed_size, device=device)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model = embed_size,
            nhead = num_heads,
            dim_feedforward = ff_hidden,
            dropout = dropout,
            batch_first=True,
            device=device
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        # Final output projection
        self.final_linear = nn.Linear(embed_size, embed_size,device=device)
        self.edge_predictor = EdgeMapping(device=device)

    def forward(self, x):
        x_emb = self.card_value_embedding(x[:,:,0] - 1) + self.card_suite_embedding(x[:,:,1] - 1)
        # Decoder forward pass
        out = self.transformer(
            src=x_emb)  # [seq_len, batch_size, embed_size]
        node_embeddings = self.final_linear(out)  # [batch_size, seq_len, vocab_size]
        return self.edge_predictor(node_embeddings)
