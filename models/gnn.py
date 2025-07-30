import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .conv import GNN_node, GNN_node_Virtualnode

from torch.distributions import Normal, Independent
from torch.nn.functional import softplus

import itertools
from typing import List, Tuple

import numpy as np
from copy import deepcopy

class GNN(torch.nn.Module):
    def __init__(
        self,
        num_tasks=None, # to remove
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        drop_ratio=0.5,
        graph_pooling="max",
        norm_layer="batch_norm",
        decoder_dims=[1024, 1111, 862, 1783, 966, 978],
        pro_dims=[1024 + 167, 300, 512, 300],
        #decoder_dims=[1024],
        # mol, gene (gc, go), cell (bray, jump), express
    ):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        ### GNN to generate node embeddings
        gnn_name = gnn_type.split("-")[0]
        if "virtual" in gnn_type:
            self.graph_encoder = GNN_node_Virtualnode(
                num_layer,
                emb_dim ,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        else:
            self.graph_encoder = GNN_node(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        
        ### Poolinwg function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        

        ######################################################################
        ### GNN to generate node embeddings
        gnn_name = gnn_type.split("-")[0]
        if "virtual" in gnn_type:
            self.biograph_encoder = GNN_node_Virtualnode(
                num_layer,
                emb_dim ,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        else:
            self.biograph_encoder = GNN_node(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        
        ### Poolinwg function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.biopool = global_add_pool
        elif graph_pooling == "mean":
            self.biopool = global_mean_pool
        elif graph_pooling == "max":
            self.biopool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        ######################################################################
        self.dist_net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * emb_dim, bias=True)
        )


        self.modal_projectors = nn.ModuleList([
            nn.Sequential(
                nn.InstanceNorm1d(pro_dim),  
                nn.Linear(pro_dim, emb_dim),
                nn.SiLU()
            ) for pro_dim in pro_dims
        ])

        self.decoder_list = nn.ModuleList()
        for out_dim in decoder_dims:
            self.decoder_list.append(MLP(emb_dim, hidden_features=emb_dim*4, out_features=out_dim))
        self.BayesOTTreeVQ = BayesOTTreeVQ(codebook_sizes=[32, 64, 128, 256], embedding_dim=emb_dim)

        



    def forward(self, batched_data):
        #pro_dims=[1024, 300, 512, 300],
        input1D = batched_data.feature_1D
        h_node, _ = self.graph_encoder(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        input2D = h_graph
        input3D = batched_data.feature_3D

        h_node_bio, _ = self.biograph_encoder(batched_data)
        h_graphbio = self.biopool(h_node_bio, batched_data.batch)
        inputs = [input1D, input2D, input3D, h_graphbio]
        inputs = [x.float() for x in inputs]

        projected_features = [projector(x) for projector, x in zip(self.modal_projectors, inputs)] #【B,D】

        idx, z, loss_rout_graph = self.BayesOTTreeVQ(projected_features)

    
        out = []
        # p, mol, gene (gc, go), cell (bray, jump), express
        for decoder in self.decoder_list:
            out.append(decoder(projected_features[3]))
        out_gene = torch.cat((out[1], out[2]), dim=1)
        out_cell = torch.cat((out[3], out[4]), dim=1)
        return [out[0], out_gene, out_cell, out[5]], loss_rout_graph
    

    def load_pretrained_graph_encoder(self, model_path):
        saved_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        graph_encoder_state_dict = {key: value for key, value in saved_state_dict.items() if key.startswith('graph_encoder.')}
        graph_encoder_state_dict = {key.replace('graph_encoder.', ''): value for key, value in graph_encoder_state_dict.items()}
        self.graph_encoder.load_state_dict(graph_encoder_state_dict)
        self.freeze_graph_encoder()

    def freeze_graph_encoder(self):
        for param in self.graph_encoder.parameters():
            param.requires_grad = False


# define a new finetune model with the same architecture of GNN with a new MLP

class FineTuneGNN(nn.Module):
    def __init__(
        self,
        num_tasks=None,
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        drop_ratio=0.5,
        graph_pooling="max",
        norm_layer="batch_norm",
    ):
        super(FineTuneGNN, self).__init__()

        ### GNN to generate node embeddings
        gnn_name = gnn_type.split("-")[0]
        if "virtual" in gnn_type:
            self.graph_encoder = GNN_node_Virtualnode(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        else:
            self.graph_encoder = GNN_node(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        ### Poolinwg function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        
        self.dist_net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * emb_dim, bias=True)
        )

        pro_dims=[1024 + 167, 300, 512, 300]
        self.modal_projectors = nn.ModuleList([
            nn.Sequential(
                nn.InstanceNorm1d(pro_dim),  
                nn.Linear(pro_dim, emb_dim),
                nn.SiLU()
            ) for pro_dim in pro_dims
        ])

        # self.task_decoder = nn.Linear(emb_dim, num_tasks)
        self.task_decoder = MLP(1024+167+3*emb_dim, hidden_features= 6 * emb_dim, out_features=num_tasks)#1024+167+emb_dim+512
        self.BayesOTTreeVQ = BayesOTTreeVQ(codebook_sizes=[64, 128, 256], embedding_dim=emb_dim)

    
    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        #projected_features = [projector(x) for projector, x in zip(self.modal_projectors, inputs)]
        projected_features_1D = self.modal_projectors[0](batched_data.mol_features.float())
        projected_features_2D = self.modal_projectors[1](h_graph)
        projected_features_3D = self.modal_projectors[2](batched_data.unimol_features)
        # mu, _ = self.dist_net(h_graph).chunk(2, dim=1)
        #idx, z, loss_rout_graph = self.BayesOTTreeVQ([projected_features_1D, projected_features_2D, projected_features_3D])
        

        task_out = self.task_decoder(torch.cat([batched_data.mol_features.float(), projected_features_1D, projected_features_2D, projected_features_3D], dim=1))# + 0.01*batched_data.rf_pred.detach()
        #task_out= self.task_decoder(torch.cat([batched_data.mol_features.float(), h_graph, batched_data.unimol_features], dim=1))

        #task_out = self.task_decoder(batched_data.unimol_features)
        return task_out#, torch.tensor(0)#100*loss_rout_graph

    def load_pretrained_graph_encoder(self, model_path):
        saved_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        graph_encoder_state_dict = {key: value for key, value in saved_state_dict.items() if key.startswith('biograph_encoder.')}
        graph_encoder_state_dict = {key.replace('biograph_encoder.', ''): value for key, value in graph_encoder_state_dict.items()}
        self.graph_encoder.load_state_dict(graph_encoder_state_dict)
        # Load dist_net state dictionary
        modal_projectors_state_dict = {key: value for key, value in saved_state_dict.items() if key.startswith('modal_projectors.')}
        modal_projectors_state_dict = {key.replace('modal_projectors.', ''): value for key, value in modal_projectors_state_dict.items()}
        self.modal_projectors.load_state_dict(modal_projectors_state_dict)
        #self.freeze_graph_encoder()

    def freeze_graph_encoder(self):
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
        for param in self.modal_projectors.parameters():
            param.requires_grad = False

class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.8,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)      # 层归一化适应分子特征分布[1,4](@ref)
        self.in1 = nn.InstanceNorm1d(in_features)  
        self.ln2 = nn.LayerNorm(out_features) 
        self.act2 = nn.SiLU()                    # 平滑激活函数提升梯度流[4,7](@ref)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.in1(x)
        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        #x = self.ln2(x)
        #x = self.drop2(x)
        return x
    
# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def sample_codebook(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """Re‑parameterised sampling from a diagonal Gaussian N(mu, sigma^2)."""
    eps = torch.randn_like(mu)
    return mu + torch.exp(log_sigma) * eps


def kl_gaussian(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """KL divergence between N(mu, sigma^2) and N(0, I) (scalar)."""
    return 0.5 * torch.sum(mu.pow(2) + torch.exp(2 * log_sigma) - 1 - 2 * log_sigma)


def sinkhorn(cost: torch.Tensor,
             mu: torch.Tensor,
             nu: torch.Tensor,
             eps: float = 0.1,
             n_iter: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Sinkhorn‑Knopp optimal transport plan and cost.

    Args:
        cost: (K1,K2) pairwise cost matrix.
        mu:   (K1,) source histogram (sums to 1).
        nu:   (K2,) target histogram (sums to 1).
    Returns:
        transport plan T (K1,K2) and OT cost (scalar).
    """
    K = torch.exp(-cost / eps)  # Gibbs kernel
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    # Add epsilon to avoid divide‑by‑zero
    mu = mu + 1e-8
    nu = nu + 1e-8

    for _ in range(n_iter):
        u = mu / (K @ v)
        v = nu / (K.t() @ u)
    T = torch.diag(u) @ K @ torch.diag(v)
    ot_cost = torch.sum(T * cost)
    return T, ot_cost

def cal_distance_matrix_with_tree(
    rooter_input,
    emb_level_item,
    last_tree_node_idx=None,
    tree_rout_bool=False,
):

    batch_size = rooter_input.shape[0]
    distances = (
        (rooter_input**2).sum(dim=1, keepdim=True)
        + (emb_level_item**2).sum(dim=1)
        - 2 * torch.matmul(rooter_input, emb_level_item.t())
    )
    if last_tree_node_idx is not None and tree_rout_bool:
        distances_plus = torch.full_like(distances, float("inf"))

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

def align_loss(rooter_input, emb_level_item, distances, temperature=0.1):
    """
    rooter_input: shape [B, D]
    emb_level_item: shape [K, D]
    distances: shape [B, K] or [K, B]
    """
    num_embeddings = emb_level_item.shape[0]

    # Step 1: 获取最近 embedding 的索引（保证是一维 long）
    encoding_indices = torch.argmin(distances, dim=1)  # shape [B]
    encoding_indices = encoding_indices.view(-1).long()

    # Step 2: One-hot 编码（[B, K]）
    encodings = torch.zeros(encoding_indices.size(0), num_embeddings, device=rooter_input.device)
    encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

    # Step 3: 量化
    quantized = torch.matmul(encodings.detach(), emb_level_item)  # shape [B, D]
    quantized = quantized.contiguous()

    # similarity = -distances / temperature
    # soft_weights = F.softmax(similarity, dim=1)  # [B, K]
    # quantized = soft_weights @ emb_level_item  # [B, D]

    # Step 4: VQ loss
    e_latent_loss = F.mse_loss(quantized.detach(), rooter_input)
    q_latent_loss = F.mse_loss(quantized, rooter_input.detach())
    loss = q_latent_loss + e_latent_loss

    # Step 5: Straight-through
    quantized = rooter_input + (quantized - rooter_input).detach()

    return encoding_indices, quantized, loss

def info_nce(z_child: torch.Tensor,
             z_parent: torch.Tensor,
             temperature: float = 0.07) -> torch.Tensor:
    """Cross‑layer InfoNCE loss (child acts as query, parent as positive)."""
    zc = F.normalize(z_child, dim=-1)
    zp = F.normalize(z_parent, dim=-1)
    logits = torch.matmul(zc, zp.t()) / temperature  # (B,B)
    labels = torch.arange(z_child.size(0), device=z_child.device)
    return F.cross_entropy(logits, labels)

# -----------------------------------------------------------------------------
# Main module: Bayesian OT Tree Vector Quantiser
# -----------------------------------------------------------------------------

class BayesOTTreeVQ(nn.Module):
    """Bayesian Optimal‑Transport Tree Vector Quantiser.

    Args:
        codebook_sizes: list of K_ℓ for each layer ℓ.
        embedding_dim:  latent dimensionality D.
        beta:   weight for KL regularisation.
        gamma:  weight for OT coupling.
        lambd:  weight for InfoNCE.
        ec_ce_weight: weight for commitment loss in VQ (kept from original design).
    """

    def __init__(self,
                 codebook_sizes: List[int],
                 embedding_dim: int,
                 beta: float = 1e-4,
                 gamma: float = 1.0,
                 lambd: float = 0.1,
                 ec_ce_weight: float = 10.0,
                 ot_eps: float = 0.1,
                 ot_iter: int = 50):
        super().__init__()
        self.depth = len(codebook_sizes)
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.gamma = gamma
        self.lambd = lambd
        self.ec_ce_weight = ec_ce_weight
        self.ot_eps = ot_eps
        self.ot_iter = ot_iter

        # Bayesian codebook parameters per layer
        self.mu_params = nn.ParameterList()
        self.logsigma_params = nn.ParameterList()
        for k in codebook_sizes:
            self.mu_params.append(nn.Parameter(torch.randn(k, embedding_dim) * 0.02))
            self.logsigma_params.append(nn.Parameter(torch.full((k, embedding_dim), -4.0)))

    # ---------------------------------------------------------------------
    # Single‑layer Vector Quantisation
    # ---------------------------------------------------------------------
    def _vq_layer(self,
                  x: torch.Tensor,
                  e: torch.Tensor,
                  temperature = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform VQ on inputs x given codebook e.

        Returns (indices, quantised_vectors, vq_loss)."""
        # Squared L2 distance (broadcasted)
        dist = (
            x.pow(2).sum(dim=1, keepdim=True) +
            e.pow(2).sum(dim=1).unsqueeze(0) -
            2 * x @ e.t()
        )  # (B,K)
        indices = torch.argmin(dist, dim=1)  # (B,)
        # enc = F.one_hot(indices, num_classes=e.size(0)).type_as(x)
        # quantised = enc @ e  # (B,D)

        similarity = -dist / temperature
        soft_weights = F.softmax(similarity, dim=1)  # [B, K]
        quantised = soft_weights @ e  # [B, D]


        # VQ losses
        e_latent = F.mse_loss(quantised.detach(), x)
        q_latent = F.mse_loss(quantised, x.detach())
        loss_vq = q_latent + 0.25 * e_latent

        # Straight‑through estimator
        quantised = x + (quantised - x).detach()
        return indices, quantised, loss_vq
    
    def vq_layer_with_tree_and_loss(self, x_l, e_l, last_tree_node_idx=None, tree_rout_bool=False, ec_ce_weight=1.0):
        """
        替代原来的 _vq_layer(x, e)
        支持树筛选、对称loss
        返回: indices, quantised, vq_loss
        """
        # Step 1: 计算距离（含树结构筛选）
        distances, distances_on_tree = cal_distance_matrix_with_tree(
            x_l, e_l, last_tree_node_idx, tree_rout_bool
        )

        # Step 2: 对称 VQ loss（x->e 和 e->x），注意输入顺序
        if last_tree_node_idx is not None:
            idx_l, z_l, loss_ec = align_loss(x_l, e_l, distances_on_tree)
            _, _, loss_ce = align_loss(e_l, x_l, distances_on_tree.t())
        else:
            idx_l, z_l, loss_ec = align_loss(x_l, e_l, distances)
            _, _, loss_ce = align_loss(e_l, x_l, distances.t())

        vq_loss = loss_ec + ec_ce_weight * loss_ce
        return idx_l.view(-1), z_l, vq_loss



    # ---------------------------------------------------------------------
    # Forward pass through all layers
    # ---------------------------------------------------------------------
    def forward(self,
                latents_per_layer: List[torch.Tensor], tree_rout_bool = False, ec_ce_weight = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(latents_per_layer) == self.depth, "Depth mismatch with codebook sizes"
        B = latents_per_layer[0].size(0)
        all_indices = []
        all_quantised = []
        total_loss = 0.0

        for l in range(self.depth):
            x_l = latents_per_layer[l].view(B, self.embedding_dim)
            mu_l = self.mu_params[l]
            log_sigma_l = self.logsigma_params[l]
            e_l = sample_codebook(mu_l, log_sigma_l)  # (K_l,D)

            #Vector quantisation
            #idx_l, z_l, vq_loss_l = self._vq_layer(x_l, e_l)

            idx_l, z_l, vq_loss_l = self.vq_layer_with_tree_and_loss(
                x_l, e_l,
                last_tree_node_idx=all_indices[-1] if l > 0 else None
            )

            all_indices.append(idx_l)
            all_quantised.append(z_l)
            total_loss = total_loss + vq_loss_l

            # KL regularisation of Bayesian codebook
            kl_l = kl_gaussian(mu_l, log_sigma_l) / mu_l.size(0)
            total_loss = total_loss + self.beta * kl_l

            # OT + InfoNCE from second layer onward
            if l > 0:
                mu_prev = self.mu_params[l - 1]  # (K_prev,D)
                K_prev, K_curr = mu_prev.size(0), mu_l.size(0)

                # Empirical histograms over codewords
                indices = all_indices[l - 1]
                pi_prev = torch.bincount(indices.view(-1).long(), minlength=K_prev).float()
                pi_prev = pi_prev / pi_prev.sum()
                pi_curr = torch.bincount(idx_l, minlength=K_curr).float()
                pi_curr = pi_curr / pi_curr.sum()

                # Pairwise cost matrix between codebook means
                cost = (
                    mu_prev.pow(2).sum(dim=1, keepdim=True) +  # (K_prev,1)
                    mu_l.pow(2).sum(dim=1).unsqueeze(0) -      # (1,K_curr)
                    2 * mu_prev @ mu_l.t()                     # (K_prev,K_curr)
                )

                _, ot_loss = sinkhorn(cost, pi_prev, pi_curr, eps=self.ot_eps, n_iter=self.ot_iter)
                total_loss = total_loss + self.gamma * ot_loss

                # InfoNCE contrastive loss between parent & child quantised vectors
                nce_loss = info_nce(z_l, all_quantised[l - 1])
                total_loss = total_loss + self.lambd * nce_loss

                # print(f"Layer {l} - VQ Loss: {vq_loss_l.item()}")
                # print(f"Layer {l} - KL Loss: {kl_l.item()}")
                # print(f"Layer {l} - OT Loss: {ot_loss.item()}")
                # print(f"Layer {l} - InfoNCE Loss: {nce_loss.item()}")

        all_indices = torch.stack(all_indices, dim=1)      # (B,L)
        all_quantised = torch.stack(all_quantised, dim=1)  # (B,L,D)
        #print(f"Total Loss: {total_loss.item()}")
        return all_indices, all_quantised, total_loss
    

