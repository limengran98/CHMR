import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import torch.nn.functional as F



class MultiModalTreeVQ(nn.Module):
    """Bayesian OT-Tree VQ block.

    Accepts either
        * a list **modalities x depth** of latent tensors (shape (B,L)), or
        * a flat list of modality tensors (shape (B,L)); the same vector is
          internally broadcast to all `depth` levels.
    Returns (indices, quantised, total_loss)
    """
    def __init__(
        self,
        *,
        num_modalities: int = 6,
        depth: int = 6,
        latent_dim: int = 32,
        ec_ce_weight: float = 1.0,
        ot_levels: Optional[List[int]] = None,
        anchor_list: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.depth = depth
        self.latent_dim = latent_dim
        self._scale = 1.0
        self.ec_ce_weight = ec_ce_weight
        self.ot_levels = ot_levels or list(range(depth))

        self.anchor_list = anchor_list
        self.tree_node_embedding_network = nn.ModuleList(
            [nn.Embedding(2 ** (i + 1), 300) for i in range(self.depth)]
        )
        
        with torch.no_grad():
            for i in range(self.depth):
                self.tree_node_embedding_network[i].weight.data.normal_(0, 1)  # Initialize weights


    # ------------------------------------------------------------------
    def _broadcast_if_flat(self, latents: List) -> List[List[torch.Tensor]]:
        """If input is flat (len=modalities) broadcast to depth levels."""
        if len(latents) == self.num_modalities and isinstance(latents[0], torch.Tensor):
            return [[t]*self.depth for t in latents]  # copy refs, OK (no in‑place)
        return latents  # assume already modal×depth



    def cal_distance_matrix_with_tree(
        self,
        rooter_input,
        emb_level_item,
        last_tree_node_idx=None,
        tree_rout_bool=False,
    ):
        # L2归一化
        rooter_input = F.normalize(rooter_input, p=2, dim=1)
        emb_level_item = F.normalize(emb_level_item, p=2, dim=1)

        # cosine similarity -> cosine distance = 1 - sim
        cosine_sim = torch.matmul(rooter_input, emb_level_item.t())  # [B, N]
        distances = 1 - cosine_sim  # cosine distance ∈ [0, 2]

        if last_tree_node_idx is not None and tree_rout_bool:
            distances_plus = torch.full_like(distances, float("inf"))

            batch_size = rooter_input.shape[0]
            row_indices = torch.arange(
                batch_size, device=rooter_input.device
            ).repeat_interleave(2)
            index_s = last_tree_node_idx * 2
            col_indices = torch.arange(2, device=rooter_input.device).repeat(
                batch_size
            ) + index_s.repeat_interleave(2)
            distances_plus[row_indices, col_indices] = 0
            distances_on_tree = distances + distances_plus
        else:
            distances_on_tree = distances

        return distances, distances_on_tree

    def align_loss(
        self,
        rooter_input,
        emb_level_item,
        distances,
        weight_e_latent=1
    ):

        num_embeddings = emb_level_item.shape[0]
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # (B*H*W, 1)

        encodings = torch.zeros(
            encoding_indices.size(0), num_embeddings, device=rooter_input.device
        )
        encodings.scatter_(1, encoding_indices, 1)  # One-hot encoding

        # Quantize and reshape
        quantized = torch.matmul(encodings.detach(), emb_level_item).view(
            rooter_input.shape
        )  # Reshape back
        quantized = quantized.contiguous()  # (B, C, H, W)

        # e_latent_loss = F.mse_loss(quantized.detach(), rooter_input)
        # q_latent_loss = F.mse_loss(quantized, rooter_input.detach())

        e_latent_loss = 1 - F.cosine_similarity(quantized.detach(), rooter_input, dim=1).mean()
        q_latent_loss = 1 - F.cosine_similarity(quantized, rooter_input.detach(), dim=1).mean()

        loss = q_latent_loss + weight_e_latent * e_latent_loss

        quantized = (
            rooter_input + (quantized - rooter_input).detach()
        )  # Straight-through estimator

        return encoding_indices, quantized, loss

    def get_tree_node_embedding(self, level: int) -> nn.Embedding:
        
        emb = self.tree_node_embedding_network[level].weight
        emb = F.normalize(emb, dim=-1)
        return emb
    
    def normalize_tree_node_embedding(self):
        """Normalize the weights of the tree node embedding network."""
        
        for i in range(self.depth):
            emb = self.tree_node_embedding_network[i].weight
            emb = F.normalize(emb, dim=-1)
            self.tree_node_embedding_network[i].weight = nn.Parameter(emb)



    def single_tree_alignment_loss(
        self, 
        rooter_input, 
        tree_rout_bool=False, 
        ec_ce_weight=1.0):
        
        tree_rout_list = []
        vector_list = []
        loss_list = []

        for i in range(self.depth):
            emb_level_item = self.get_tree_node_embedding(i)
            if i > 0:
                last_tree_node_idx = tree_rout_list[-1]
            else:
                last_tree_node_idx = None

            distances, distances_on_tree = self.cal_distance_matrix_with_tree(
                rooter_input, emb_level_item, last_tree_node_idx, tree_rout_bool
            )

            if last_tree_node_idx is not None:
                encoding_indices, quantized, loss_ec_tree = self.align_loss(
                    rooter_input, emb_level_item, distances_on_tree
                )
                _, _, loss_ce_tree = self.align_loss(
                    emb_level_item, rooter_input, distances_on_tree.t()
                )
                loss = loss_ec_tree  + loss_ce_tree * ec_ce_weight
            else:
                encoding_indices, quantized, loss_ec = self.align_loss(
                    rooter_input, emb_level_item, distances
                )
                _, _, loss_ce = self.align_loss(
                    emb_level_item, rooter_input, distances.t()
                )
                loss = loss_ec  + loss_ce * ec_ce_weight

            tree_rout_list.append(encoding_indices.reshape(-1))
            vector_list.append(quantized)
            loss_list.append(loss)

        tree_rout = torch.stack(tree_rout_list, axis=1)
        vector_rout = torch.stack(vector_list, axis=1)
        loss = torch.stack(loss_list).mean()
        return tree_rout, vector_rout, loss


    # ------------------------------------------------------------------
    def forward(self, latents_in: List) -> Tuple[List[List[torch.Tensor]],
                                                List[List[torch.Tensor]],
                                                torch.Tensor]:
        """Return (all_indices, all_quantised, total_loss)."""

        loss_rout_list = []
        tree_rout_dict = {}
        vector_rout_dict = {}

        for m in range(self.num_modalities):
            x = latents_in[m]
            tree_rout, vector_rout, loss_rout = self.single_tree_alignment_loss(
                rooter_input=x,
                tree_rout_bool=True,
                ec_ce_weight=self.ec_ce_weight,
            )
            weight = 1.0
            loss_rout_weighted = loss_rout * weight

            tree_rout_dict[f'm{m}'] = tree_rout
            vector_rout_dict[f'm{m}'] = vector_rout


            loss_rout_list.append(loss_rout_weighted)

        total_loss = torch.stack(loss_rout_list).mean()

        return tree_rout_dict, vector_rout_dict, total_loss


