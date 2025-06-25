import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .conv import GNN_node, GNN_node_Virtualnode

from typing import List, Tuple

from torch_cluster import random_walk



def _bce_with_mask(pred, target, weight, nan_mask):
    """Binary‑cross‑entropy that skips NaNs and supports per‑step weights."""
    expanded_weight = weight.unsqueeze(2).expand_as(target)
    loss = F.binary_cross_entropy_with_logits(
        pred[~nan_mask],
        target[~nan_mask],
        weight=expanded_weight[~nan_mask],
        reduction="none",
    )
    return loss.mean()



def pretrain_func(args, model: nn.Module, loaders: dict, ctx_graph, optimizer, scheduler, epoch):
    """One pre‑training epoch with random‑walk context supervision."""

    model.train()
    device = args.device

    # ---- helpers & meters --------------------------------------------------
    from utils.misc import AverageMeter  # deferred import to keep canvas self‑contained
    meters = {k: AverageMeter() for k in ["tot", "route", "mol", "gene", "cell", "expr"]}

    # ---- constants from context graph -------------------------------------
    edge_w = torch.cat([ctx_graph.weight,
                        torch.zeros(1, dtype=ctx_graph.weight.dtype, device=ctx_graph.weight.device)])
    idx_map = {t: i for i, t in enumerate(ctx_graph.type)}

    # epoch‑level progress bar
    pbar = tqdm(range(args.steps), disable=args.no_print)

    for step in pbar:
        tic = time.time()
        optimizer.zero_grad(set_to_none=True)


        # global_step = epoch * args.steps + step
        # scale = min(1.0, global_step / max(1, args.warmup_steps))
        # model.BayesTreeVQ.set_warmup_scale(scale)


        # ----- fetch batch ---------------------------------------------------
        try:
            batch = next(loaders["train_iter"])
        except StopIteration:
            loaders["train_iter"] = iter(loaders["train_loader"])
            batch = next(loaders["train_iter"])
        batch = batch.to(device)
        if batch.x.size(0) == 1 or batch.batch[-1] == 0:  # skip trivial graphs
            continue

        # ----- random walk on heterogeneous context graph -------------------
        start_idx = torch.tensor([idx_map[t] for t in batch.type])
        walk, edge_seq = random_walk(ctx_graph.edge_index[0], ctx_graph.edge_index[1],
                                     start_idx, args.walk_length,
                                     num_nodes=ctx_graph.num_nodes, return_edge_indices=True)
        path_w = edge_w[edge_seq].view(-1, args.walk_length).cumprod(-1).to(device)
        path_w = torch.cat([torch.ones_like(path_w[:, :1]), path_w], dim=1)  # include start step

        # ----- gather targets ------------------------------------------------
        tgt_mol = ctx_graph.mol_target[walk].to(device)
        tgt_gene = ctx_graph.gene_target[walk].to(device)
        tgt_cell = ctx_graph.cell_target[walk].to(device)
        tgt_expr = ctx_graph.express_target[walk].to(device)
        masks = {"mol": torch.isnan(tgt_mol),
                 "gene": torch.isnan(tgt_gene),
                 "cell": torch.isnan(tgt_cell),
                 "expr": torch.isnan(tgt_expr)}

        # ----- forward -------------------------------------------------------
        preds, loss_route = model(batch)
        pred_mol, pred_gene, pred_cell, pred_expr = preds

        # broadcast preds to (B, L, C)
        expand = lambda p, tgt: p.unsqueeze(1).expand(-1, tgt.size(1), -1)
        p_mol, p_gene, p_cell, p_expr = [expand(p, t) for p, t in zip(preds, [tgt_mol, tgt_gene, tgt_cell, tgt_expr])]

        # ----- losses --------------------------------------------------------
        loss_mol = _bce_with_mask(p_mol, tgt_mol, path_w, masks["mol"])
        loss_gene = _bce_with_mask(p_gene, tgt_gene, path_w, masks["gene"])
        loss_cell = _bce_with_mask(p_cell, tgt_cell, path_w, masks["cell"])
        loss_expr = _bce_with_mask(p_expr, tgt_expr, path_w, masks["expr"])
        #loss_expr = torch.nan_to_num(loss_expr, nan=0.0)
        loss_tot = loss_mol + loss_gene + loss_cell + loss_expr + args.prior * loss_route

        # ----- backward & step ---------------------------------------------
        loss_tot.backward()
        # if args.clip_grad > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        scheduler.step()

        # ----- logging -------------------------------------------------------
        meters["tot"].update(loss_tot.item())
        meters["route"].update((args.prior * loss_route).item())
        meters["mol"].update(loss_mol.item())
        meters["gene"].update(loss_gene.item())
        meters["cell"].update(loss_cell.item())
        meters["expr"].update(loss_expr.item())

        pbar.set_description(
            f"Ep {epoch+1}/{args.epochs} | It {step+1}/{args.steps} | "
            f"Loss {meters['tot'].avg:.3f} (M {meters['mol'].avg:.3f} G {meters['gene'].avg:.3f} "
            f"C {meters['cell'].avg:.3f} E {meters['expr'].avg:.3f} R {meters['route'].avg:.3f})")

    pbar.close()
    if not args.no_print:
        args.logger.info(f"Epoch {epoch+1}: Total {meters['tot'].avg:.4f}")
    return meters["tot"].avg, loaders


class GNN(nn.Module):
    """Graph encoder + Bayes OT‑Tree VQ + multi‑head decoders."""

    def __init__(self, *, num_layer: int = 5, emb_dim: int = 300,
                 gnn_type: str = "gin", drop_ratio: float = 0.5,
                 graph_pooling: str = "max", norm_layer: str = "batch_norm",
                 decoder_dims: list[int] | None = None,
                 pro_dims: list[int] | None = None,
                 depth: int = 6,
                 beta: float = 1.0,
                 gamma: float = 1.0,
                 lambd: float = 1.0,
                 ec_ce_weight: float = 1.0,
                 return_tree=False):
        super().__init__()
        decoder_dims = decoder_dims or [1024, 1111, 862, 1783, 966, 978]
        pro_dims = pro_dims or [1024 + 167, 300, 512, 300]

        # -------- graph encoders (structural + bio) --------
        gnn_name = gnn_type.split("-")[0]
        node_cls = GNN_node_Virtualnode if "virtual" in gnn_type else GNN_node  # type: ignore
        self.graph_encoder = node_cls(num_layer, emb_dim, JK="last", drop_ratio=drop_ratio,
                                      residual=True, gnn_name=gnn_name, norm_layer=norm_layer)
        self.biograph_encoder = node_cls(num_layer, emb_dim, JK="last", drop_ratio=drop_ratio,
                                         residual=True, gnn_name=gnn_name, norm_layer=norm_layer)

        pool_map = {"sum": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}
        if graph_pooling not in pool_map:
            raise ValueError("Invalid graph pooling type")
        self.pool = pool_map[graph_pooling]
        self.biopool = pool_map[graph_pooling]

        # -------- modality projectors 1D/2D/3D/CTX --------
        self.modal_projectors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),        
                nn.Linear(dim, emb_dim),
                nn.SiLU()                
            )
            for dim in pro_dims
        ])

        # -------- per‑task decoders --------
        self.decoder_list = nn.ModuleList([
            MLP(emb_dim, hidden_features=emb_dim * 4, out_features=d) for d in decoder_dims
        ])
        # self.decoder_list = nn.ModuleList([
        #     nn.Sequential(
        #         nn.LayerNorm(emb_dim),        
        #         nn.Linear(emb_dim, d),
        #         nn.SiLU()                
        #     )
        #     for d in decoder_dims
        # ])

        # -------- OT‑Tree VQ block --------
        self.BayesTreeVQ = MultiModalTreeVQ(depth=depth, latent_dim=emb_dim, beta=beta, gamma=gamma, lambd=lambd, ec_ce_weight=ec_ce_weight)
        self.return_tree = return_tree
    # ------------------------------------------------------------------
    def forward(self, data):
        # Assume `data` has attributes: feature_1D, feature_3D, and is a PyG Batch
        input1D = data.feature_1D.float()
        h_node = self.graph_encoder(data)[0]
        input2D = self.pool(h_node, data.batch)
        input3D = data.feature_3D.float()
        h_node_bio = self.biograph_encoder(data)[0]
        ctx = self.biopool(h_node_bio, data.batch)

        # projector outputs (B, emb_dim)
        projected = [proj(t) for proj, t in zip(self.modal_projectors, [input1D, input2D, input3D, ctx])]

        # OT‑Tree routing + loss
        all_idx, all_z, loss_rout = self.BayesTreeVQ(projected)

        # heads
        out = [dec(projected[3]) for dec in self.decoder_list]
        out_gene = torch.cat((out[1], out[2]), dim=1)
        out_cell = torch.cat((out[3], out[4]), dim=1)
        predictions = [out[0], out_gene, out_cell, out[5]]

        raw_data = [input1D, input2D, input3D, ctx]
        if self.return_tree == True:
            return all_idx, all_z, projected, out, raw_data
        else:
            return predictions, loss_rout
    # ---- utility: load & freeze graph encoder ----
    def load_pretrained_graph_encoder(self, path: str):
        state = torch.load(path, map_location="cpu")
        ge_state = {k.replace("graph_encoder.", ""): v for k, v in state.items() if k.startswith("graph_encoder.")}
        self.graph_encoder.load_state_dict(ge_state)
        self.freeze_graph_encoder()

    def freeze_graph_encoder(self):
        for p in self.graph_encoder.parameters():
            p.requires_grad_(False)


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None,
                 out_features: int | None = None, act_layer: type[nn.Module] = nn.SiLU,
                 bias: bool = True, drop: float = 0.3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features, bias=bias),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    
# -----------------------------------------------------------------------------
# 1.  GaussianTreeModel  (unchanged)
# -----------------------------------------------------------------------------

class GaussianTreeModel(nn.Module):
    """Balanced binary‑tree codebook with Gaussian nodes."""

    def __init__(self, *, num_modalities: int, depth: int, latent_dim: int):
        super().__init__()
        self.num_modalities = num_modalities
        self.depth = depth
        self.latent_dim = latent_dim

        self.modal_trees = nn.ModuleList([
            nn.ModuleList([
                nn.Embedding(2 ** d, 2 * latent_dim)
                for d in range(depth)
            ]) for _ in range(num_modalities)
        ])

    @staticmethod
    def _kl_unit_gaussian(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)

    def forward(self) -> Tuple[List[List[torch.Tensor]], torch.Tensor]:
        all_mus: List[List[torch.Tensor]] = []
        total_kl = torch.tensor(0.0, device=next(self.parameters()).device)

        for m in range(self.num_modalities):
            mus_m = []
            for d, layer in enumerate(self.modal_trees[m]):
                K = 2 ** d
                emb = layer(torch.arange(K, device=layer.weight.device))
                mu, logvar = torch.chunk(emb, 2, dim=-1)
                mus_m.append(mu)
                total_kl = total_kl + self._kl_unit_gaussian(mu, logvar)
            all_mus.append(mus_m)

        total_nodes = sum(2 ** d for d in range(self.depth)) * self.num_modalities
        kl_loss = total_kl / (total_nodes * self.latent_dim)
        return all_mus, kl_loss


# -----------------------------------------------------------------------------
# 2.  Tree‑Wasserstein OT & InfoNCE (unchanged)
# -----------------------------------------------------------------------------

def tree_wasserstein(pi1: torch.Tensor, pi2: torch.Tensor, *, edge_length: float = 1.0) -> torch.Tensor:
    pi1 = pi1 / (pi1.sum() + 1e-8)
    pi2 = pi2 / (pi2.sum() + 1e-8)
    dist = torch.tensor(0.0, device=pi1.device)
    h1, h2 = pi1.clone(), pi2.clone()
    while h1.numel() > 1:
        dist = dist + (h1 - h2).abs().sum() * edge_length
        h1 = h1.view(-1, 2).sum(dim=1)
        h2 = h2.view(-1, 2).sum(dim=1)
    return dist


def info_nce(z_a: torch.Tensor, z_b: torch.Tensor, *, temperature: float = 0.07) -> torch.Tensor:
    za = F.normalize(z_a, dim=-1)
    zb = F.normalize(z_b, dim=-1)
    logits = za @ zb.t() / temperature
    labels = torch.arange(za.size(0), device=za.device)
    return F.cross_entropy(logits, labels)


# -----------------------------------------------------------------------------
# 3.  Vector‑Quantisation with Tree Routing  (Version‑1 kept)
# -----------------------------------------------------------------------------

def _cal_distance_matrix_with_tree(x: torch.Tensor,
                                   codebook: torch.Tensor,
                                   last_idx: torch.Tensor = None,
                                   tree_route: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return full distance and tree‑masked distance matrices."""
    dist = (
        x.pow(2).sum(dim=1, keepdim=True) +
        codebook.pow(2).sum(dim=1).unsqueeze(0) -
        2 * x @ codebook.t()
    )  # (B,K)
    if tree_route and (last_idx is not None):
        B, K = dist.shape
        dist_masked = torch.full_like(dist, float('inf'))
        row_idx = torch.arange(B, device=x.device).repeat_interleave(2)
        col_idx = last_idx.long().repeat_interleave(2) * 2 + torch.arange(2, device=x.device).repeat(B)
        dist_masked[row_idx, col_idx] = 0.0
        dist_tree = dist + dist_masked
    else:
        dist_tree = dist
    return dist, dist_tree


def _align_loss(query: torch.Tensor, keybook: torch.Tensor, dist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K = keybook.size(0)
    indices = torch.argmin(dist, dim=1)
    enc = F.one_hot(indices, num_classes=K).type_as(query)
    quantised = enc @ keybook
    e_latent = F.mse_loss(quantised.detach(), query)
    q_latent = F.mse_loss(quantised, query.detach())
    loss = e_latent + q_latent
    quantised = query + (quantised - query).detach()  # straight‑through
    return indices, quantised, loss


def vq_layer_with_tree_and_loss(x: torch.Tensor,
                                codebook: torch.Tensor,
                                *,
                                last_idx: torch.Tensor = None,
                                tree_route: bool = False,
                                ec_ce_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard‑assignment VQ with optional tree routing and symmetric loss."""
    full_dist, tree_dist = _cal_distance_matrix_with_tree(x, codebook, last_idx, tree_route)
    # encoder→codebook (EC)
    idx, z, loss_ec = _align_loss(x, codebook, tree_dist)
    # codebook→encoder (CE) for commitment/symmetry
    _, _, loss_ce = _align_loss(codebook, x, tree_dist.t())
    return idx, z, loss_ec + ec_ce_weight * loss_ce


# -----------------------------------------------------------------------------
# 4.  Hierarchical alignment loss (unchanged)
# -----------------------------------------------------------------------------

def hierarchical_alignment_loss(all_indices: List[List[torch.Tensor]],
                                 all_quantised: List[List[torch.Tensor]],
                                 *,
                                 depth: int,
                                 num_modalities: int,
                                 ot_levels: List[int],
                                 gamma: float,
                                 lambd: float,
                                 use_ot: bool = True,
                                 use_nce: bool = True) -> torch.Tensor:
    loss = torch.tensor(0.0, device=all_quantised[0][0].device)
    for d in range(depth):
        for m in range(num_modalities - 1):
            n = m + 1
            idx_m, idx_n = all_indices[m][d], all_indices[n][d]
            z_m, z_n = all_quantised[m][d], all_quantised[n][d]
            if use_ot and (d in ot_levels):
                K = 2 ** d
                pi_m = torch.bincount(idx_m.long(), minlength=K).float()
                pi_n = torch.bincount(idx_n.long(), minlength=K).float()
                loss = loss + gamma * tree_wasserstein(pi_m, pi_n)
            if use_nce:
                loss = loss + lambd * info_nce(z_m, z_n)
    return loss


# -----------------------------------------------------------------------------
# 5.  Main model tying everything together
# -----------------------------------------------------------------------------

class MultiModalTreeVQ(nn.Module):
    """Bayesian OT‑Tree VQ block.

    Accepts either
        * a list **modalities x depth** of latent tensors (shape (B,L)), or
        * a flat list of modality tensors (shape (B,L)); the same vector is
          internally broadcast to all `depth` levels.
    Returns (indices, quantised, total_loss)
    """
    def __init__(self, *, num_modalities: int = 4, depth: int = 5, latent_dim: int = 32,
                 beta: float = 1.0, gamma: float = 1.0, lambd: float = 1.0,
                 ec_ce_weight: float = 1.0,
                 ot_levels: List[int] = None):
        super().__init__()
        self.num_modalities = num_modalities
        self.depth = depth
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.lambd = lambd
        self._scale = 1.0
        self.ec_ce_weight = ec_ce_weight
        self.ot_levels = ot_levels or list(range(depth))
        self.tree = GaussianTreeModel(num_modalities=num_modalities,
                                      depth=self.depth,
                                      latent_dim=latent_dim)

    # ------------------------------------------------------------------
    def _broadcast_if_flat(self, latents: List) -> List[List[torch.Tensor]]:
        """If input is flat (len=modalities) broadcast to depth levels."""
        if len(latents) == self.num_modalities and isinstance(latents[0], torch.Tensor):
            return [[t]*self.depth for t in latents]  # copy refs, OK (no in‑place)
        return latents  # assume already modal×depth

    # ------------------------------------------------------------------
    def forward(self, latents_in: List) -> Tuple[List[List[torch.Tensor]],
                                                List[List[torch.Tensor]],
                                                torch.Tensor]:
        """Return (all_indices, all_quantised, total_loss)."""
        latents = self._broadcast_if_flat(latents_in)
        device = latents[0][0].device
        B = latents[0][0].size(0)
        codebook_mus, kl = self.tree()

        all_idx: List[List[torch.Tensor]] = [[None]*self.depth for _ in range(self.num_modalities)]
        all_z:   List[List[torch.Tensor]] = [[None]*self.depth for _ in range(self.num_modalities)]
        vq_loss_total = torch.tensor(0.0, device=device)

        for m in range(self.num_modalities):
            for d in range(self.depth):
                x = latents[m][d].view(B, self.latent_dim)
                e = codebook_mus[m][d]
                parent_idx = all_idx[m][d-1] if d > 0 else None
                idx, z, vq_loss = vq_layer_with_tree_and_loss(
                    x, e,
                    last_idx=parent_idx,
                    tree_route=True,
                    ec_ce_weight=self.ec_ce_weight
                )
                all_idx[m][d], all_z[m][d] = idx, z
                vq_loss_total = vq_loss_total + vq_loss

        align_loss = hierarchical_alignment_loss(all_idx, all_z,
                                                 depth=self.depth,
                                                 num_modalities=self.num_modalities,
                                                 ot_levels=self.ot_levels,
                                                 gamma=self.gamma,
                                                 lambd=self.lambd)
        total_loss = vq_loss_total + self.beta*kl + align_loss
        return all_idx, all_z, total_loss
    