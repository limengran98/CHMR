import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_cluster import random_walk

import numpy as np

from utils.plot import plot_tsne
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .tree_model import MultiModalTreeVQ


def context_propagation_recon_loss(pred, target, weight, nan_mask):
    """
    CPR Loss: Graph-propagated neighborhood features as supervision for reconstruction.
    """
    expanded_weight = weight.unsqueeze(2).expand_as(target)
    loss = F.binary_cross_entropy_with_logits(
        pred[~nan_mask],
        target[~nan_mask],
        weight=expanded_weight[~nan_mask],
        reduction="none",
    )
    return loss.mean()

def semantic_consistency_alignment_loss(
    zi: torch.Tensor,
    zj: torch.Tensor,
    temperature: float = 0.2,
    mode: str = "info_nce",  # "info_nce" or "vicreg"
    invar: float = 1.0,
    var: float = 1.0,
    cov: float = 1.0,
    eps: float = 1e-4,
):
    """
    Computes alignment loss between two projected modalities.
    Supports InfoNCE and VICReg styles.

    Args:
        zi, zj (Tensor): (B, D) projected features from two modalities
        temperature (float): temperature scaling for InfoNCE
        mode (str): "info_nce" or "vicreg"
        invar, var, cov (float): VICReg weights
        eps (float): numerical stability for std
    Returns:
        loss (Tensor): computed alignment loss
    """
    if mode == "info_nce":
        # === InfoNCE style ===
        z_a = F.normalize(zi, dim=-1)
        z_b = F.normalize(zj, dim=-1)
        logits = z_a @ z_b.t() / temperature  # (B, B)
        labels = torch.arange(z_a.size(0), device=z_a.device)
        return F.cross_entropy(logits, labels)

    elif mode == "vicreg":
        # === VICReg style ===
        # 1. Invariance loss (MSE)
        invariance_loss = F.mse_loss(zi, zj)

        # 2. Variance regularization
        def variance_loss(x):
            std = torch.sqrt(x.var(dim=0) + eps)
            return torch.mean(F.relu(1.0 - std))

        var_loss = variance_loss(zi) + variance_loss(zj)

        # 3. Covariance regularization
        def covariance_loss(x):
            x = x - x.mean(dim=0, keepdim=True)
            cov = (x.T @ x) / (x.size(0) - 1)
            cov_no_diag = cov - torch.diag_embed(torch.diagonal(cov))
            return (cov_no_diag ** 2).sum() / x.size(1)

        cov_loss = covariance_loss(zi) + covariance_loss(zj)

        return invar * invariance_loss + var * var_loss + cov * cov_loss

    else:
        raise ValueError(f"Unknown alignment mode: {mode}")




def inter_modal_hierarchy_loss(vq_loss, weight=1.0):
    """
    Tree-structured vector quantization loss for inter-modal consistency.
    """
    return weight * vq_loss



def pretrain_func(args, model, loaders, aligned_data, aug_data, ctx_graph, optimizer, scheduler, epoch):
    """
    Pre-training function for multi-modal representation learning.
    Includes:
        - Context-Propagation Reconstruction (CPR) loss
        - Intra-Modal Alignment loss
        - Inter-Modal Hierarchy (VQ) loss
    """
    model.train()
    device = args.device

    from utils.misc import AverageMeter
    meters = {k: AverageMeter() for k in ["total", "cpr", "intra", "inter"]}

    mol_id_to_idx = {k: i for i, k in enumerate(aligned_data["mol_id"])}

    # === Constants from context graph ===
    edge_w = torch.cat([ctx_graph.weight,
                        torch.zeros(1, dtype=ctx_graph.weight.dtype, device=ctx_graph.weight.device)])
    idx_map = {t: i for i, t in enumerate(ctx_graph.type)}

    # Progress bar
    pbar = tqdm(range(args.steps), disable=args.no_print)

    #model.hierarchical_vq.normalize_tree_node_embedding()

    emb_list_mo_1D = []
    emb_list_mo_2D = []
    emb_list_mo_3D = []
    emb_list_mo_anchor = []
    emb_list_cell= []
    emb_list_gene = []
    emb_list_gene_expr = []

    for step in pbar:
        optimizer.zero_grad(set_to_none=True)

        try:
            batch = next(loaders["train_iter"])
        except StopIteration:
            loaders["train_iter"] = iter(loaders["train_loader"])
            batch = next(loaders["train_iter"])
        batch = batch.to(device)

        if batch.x.size(0) == 1 or batch.batch[-1] == 0:
            continue

        

        # === Gather modality inputs ===
        input1D = batch.feature_1D.float()
        input2D = model.pool(model.graph_encoder(batch)[0], batch.batch)
        input3D = batch.feature_3D.float()

        mol_ids = batch.type
        selected_idx = [mol_id_to_idx[mid] for mid in mol_ids]

        gene_crispr_emb = torch.tensor(aligned_data["crispr_feat"][selected_idx], device=device, dtype=torch.float32)
        gene_orf_emb = torch.tensor(aligned_data["orf_feat"][selected_idx], device=device, dtype=torch.float32)
        input_gene = torch.cat([gene_crispr_emb, gene_orf_emb], dim=1)

        cell_cp_bray_emb = torch.tensor(aligned_data["cp_bray_feat"][selected_idx], device=device, dtype=torch.float32)
        cell_cp_jump_emb = torch.tensor(aligned_data["cp_jump_feat"][selected_idx], device=device, dtype=torch.float32)
        input_cell = torch.cat([cell_cp_bray_emb, cell_cp_jump_emb], dim=1)

        input_express = torch.tensor(aligned_data["express_feat"][selected_idx], device=device, dtype=torch.float32)

        modal_inputs = [input1D, input2D, input3D, input_gene, input_cell, input_express]


        gene_crispr_emb_aug = torch.tensor(aug_data["crispr_feat"][selected_idx], device=device, dtype=torch.float32)
        gene_orf_emb_aug = torch.tensor(aug_data["orf_feat"][selected_idx], device=device, dtype=torch.float32)
        input_gene_aug  = torch.cat([gene_crispr_emb_aug, gene_orf_emb_aug], dim=1)

        cell_cp_bray_emb_aug = torch.tensor(aug_data["cp_bray_feat"][selected_idx], device=device, dtype=torch.float32)
        cell_cp_jump_emb_aug = torch.tensor(aug_data["cp_jump_feat"][selected_idx], device=device, dtype=torch.float32)
        input_cell_aug = torch.cat([cell_cp_bray_emb_aug, cell_cp_jump_emb_aug], dim=1)

        input_express_aug = torch.tensor(aug_data["express_feat"][selected_idx], device=device, dtype=torch.float32)

        modal_inputs_aug = [input_gene_aug, input_cell_aug, input_express_aug]


        # === Graph random walk (for CPR supervision) ===
        start_idx = torch.tensor([idx_map[t] for t in batch.type])
        walk, edge_seq = random_walk(ctx_graph.edge_index[0], ctx_graph.edge_index[1],
                                     start_idx, args.walk_length,
                                     num_nodes=ctx_graph.num_nodes, return_edge_indices=True)
        path_w = edge_w[edge_seq].view(-1, args.walk_length).cumprod(-1).to(device)
        path_w = torch.cat([torch.ones_like(path_w[:, :1]), path_w], dim=1)  # include start step

        # === Targets from graph propagation ===

        tgt_mol = ctx_graph.mol_target[walk].to(device)
        tgt_gene = ctx_graph.gene_target[walk].to(device)
        tgt_cell = ctx_graph.cell_target[walk].to(device)
        tgt_expr = ctx_graph.express_target[walk].to(device)
        masks = {"mol": torch.isnan(tgt_mol),
                 "gene": torch.isnan(tgt_gene),
                 "cell": torch.isnan(tgt_cell),
                 "expr": torch.isnan(tgt_expr)}



        # === Forward pass ===
        latent_codes, latent_vectors, vq_loss, modal_latents, projected_aug, modal_reconstructions = model(modal_inputs, modal_inputs_aug)




        # === Compute losses ===
        # --- (1) Context-Propagation Reconstruction Loss (CPR) ---
        expand = lambda p, tgt: p.unsqueeze(1).expand(-1, tgt.size(1), -1)
        preds = [expand(p, t) for p, t in zip(modal_reconstructions, [tgt_mol, tgt_gene, tgt_cell, tgt_expr])]
        targets = [tgt_mol, tgt_gene, tgt_cell, tgt_expr]
        modality_keys = ["mol", "gene", "cell", "expr"]

        cpr_loss = 0.0
        for pred, tgt, key in zip(preds, targets, modality_keys):
            cpr_loss += context_propagation_recon_loss(pred, tgt, path_w, masks[key])

        # --- (2) Intra-Modal Alignment Loss ---
        anchor_latent = sum(modal_latents[:3])  # Combine molecular modalities
        intra_loss = 0.0
        for j in range(3, len(modal_latents)):
            intra_loss += semantic_consistency_alignment_loss(anchor_latent, modal_latents[j])
            intra_loss += semantic_consistency_alignment_loss(
            modal_latents[j],
            projected_aug[j-3],
            mode = 'vicreg'
            )

        # --- (3) Inter-Modal Hierarchical Loss (Tree VQ) ---
        inter_loss = inter_modal_hierarchy_loss(vq_loss)

        
        # --- Total Loss ---
        total_loss = (
            cpr_loss +
            args.intra_weight * intra_loss +
            args.inter_weight * inter_loss
        )

        # === Backward and optimize ===
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # === Update meters ===
        meters["total"].update(total_loss.item())
        meters["cpr"].update(cpr_loss.item())
        meters["intra"].update(intra_loss.item())
        meters["inter"].update(inter_loss.item())


        pbar.set_description(
            f"Ep {epoch+1}/{args.epochs} | It {step+1}/{args.steps} | "
            f"Tot {meters['total'].avg:.3f} "
            f"(CPR {meters['cpr'].avg:.3f}, Intra {meters['intra'].avg:.3f}, Inter {meters['inter'].avg:.3f})"
        )

    if args.vis_plot == True:
        emb_list_mo_1D.append(modal_latents[0].detach().cpu().numpy())
        emb_list_mo_2D.append(modal_latents[1].detach().cpu().numpy())
        emb_list_mo_3D.append(modal_latents[2].detach().cpu().numpy())
        emb_list_mo_anchor.append(anchor_latent.detach().cpu().numpy())
        emb_list_gene.append(projected_aug[0].detach().cpu().numpy())
        emb_list_cell.append(projected_aug[1].detach().cpu().numpy())
        emb_list_gene_expr.append(projected_aug[2].detach().cpu().numpy())

        m0_tree_rout_dict = latent_codes['m0']
        if epoch % 10 == 0:
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            
            # Concatenate the embeddings from the four lists
            emb_list_mo_1D_numpy = np.concatenate(emb_list_mo_1D, axis=0)
            emb_list_mo_2D_numpy = np.concatenate(emb_list_mo_2D, axis=0)
            emb_list_mo_3D_numpy = np.concatenate(emb_list_mo_3D, axis=0)
            emb_list_mo_anchor_numpy = np.concatenate(emb_list_mo_anchor, axis=0)
            emb_list_gene_numpy = np.concatenate(emb_list_gene, axis=0)
            emb_list_cell_numpy = np.concatenate(emb_list_cell, axis=0)
            emb_list_gene_expr_numpy = np.concatenate(emb_list_gene_expr, axis=0)


            emb_all = np.concatenate([
                emb_list_mo_1D_numpy,
                emb_list_mo_2D_numpy,
                emb_list_mo_3D_numpy,
                # emb_list_mo_anchor_numpy,
                emb_list_gene_numpy,
                emb_list_cell_numpy,
                emb_list_gene_expr_numpy
            ], axis=0)


            labels = (
                [0] * emb_list_mo_1D_numpy.shape[0] +
                [1] * emb_list_mo_2D_numpy.shape[0] +
                [2] * emb_list_mo_3D_numpy.shape[0] +
                # [0] * emb_list_mo_anchor_numpy.shape[0] +
                [3] * emb_list_gene_numpy.shape[0] +
                [4] * emb_list_cell_numpy.shape[0] +
                [5] * emb_list_gene_expr_numpy.shape[0]
            )
            labels = np.array(labels, dtype=int)

            path_dict = plot_tsne(
                emb_all=emb_all,
                # labels=np.array(labels),
                labels=labels,
                epoch=epoch+1,
                m0_tree_rout_dict=m0_tree_rout_dict,)
        


    pbar.close()
    if not args.no_print:
        args.logger.info(
            f"Epoch {epoch+1}: "
            f"Total {meters['total'].avg:.4f}, "
            f"CPR {meters['cpr'].avg:.4f}, "
            f"Intra {meters['intra'].avg:.4f}, "
            f"Inter {meters['inter'].avg:.4f}"
        )
    return meters["total"].avg, loaders


class GNN(torch.nn.Module):
    def __init__(self, *, num_layer=5, emb_dim=300, gnn_type="gin", drop_ratio=0.5,
                 graph_pooling="max", norm_layer="batch_norm",
                 pro_dims=None, decoder_dims=None, depth=6, ec_ce_weight=1.0):
        super().__init__()
        decoder_dims = decoder_dims or [1024, 1111+862, 1783+966, 978]
        pro_dims = pro_dims or [1024+167, 300, 512, 1111+862, 1783+966, 978]  # mol, gene, cell, expr dims  [300, 1111, 862, 1783, 966, 978, 1024+167, 512]

        from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
        from .conv import GNN_node, GNN_node_Virtualnode

        gnn_name = gnn_type.split("-")[0]
        node_cls = GNN_node_Virtualnode if "virtual" in gnn_type else GNN_node

        self.graph_encoder = node_cls(num_layer, emb_dim, JK="last", drop_ratio=drop_ratio,
                                      residual=True, gnn_name=gnn_name, norm_layer=norm_layer)

        pool_map = {"sum": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}
        if graph_pooling not in pool_map:
            raise ValueError("Invalid graph pooling type")
        self.pool = pool_map[graph_pooling]

        self.modal_projectors = torch.nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),        
                nn.Linear(dim, emb_dim),
                nn.SiLU()                
            ) for dim in pro_dims
        ])


        self.modal_decoders = nn.ModuleList([
            MLP(emb_dim, hidden_features=emb_dim * 4, out_features=d) for d in decoder_dims
        ])

        self.hierarchical_vq = MultiModalTreeVQ(num_modalities = len(pro_dims), depth=depth, latent_dim=emb_dim, ec_ce_weight=ec_ce_weight, anchor_list = [0, 1, 2])

    def forward(self, modal_inputs, modal_inputs_aug=None):
        """
        Args:
            modal_inputs: List of tensors for each modality.
        Returns:
            - latent_codes: Hierarchical latent indices per modality
            - modal_latents: Projected modal features
            - modal_reconstructions: Decoded modal outputs
        """
        # Project modalities into shared embedding space
        modal_latents = [proj(x) for proj, x in zip(self.modal_projectors, modal_inputs)]

        projected_aug = None
        if modal_inputs_aug is not None:
            projected_aug = [
                proj(x_aug) for proj, x_aug in zip(self.modal_projectors[3:], modal_inputs_aug)
            ]

        # Aggregate molecular modalities as anchor
        anchor_latent = sum(modal_latents[:3])  # Combine 1D, 2D, 3D features

        # Decode modalities
        modal_reconstructions = [
            decoder(x) for decoder, x in zip(
                self.modal_decoders,
                [anchor_latent, modal_latents[3], modal_latents[4], modal_latents[5]]
            )
        ]

        # Hierarchical vector quantization
        latent_codes, latent_vectors, vq_loss = self.hierarchical_vq(modal_latents)

        return latent_codes, latent_vectors, vq_loss, modal_latents, projected_aug, modal_reconstructions


    # # ---- utility: load & freeze graph encoder ----
    def load_pretrained_graph_encoder(self, path: str):
        state = torch.load(path, map_location="cpu")
        ge_state = {k.replace("graph_encoder.", ""): v for k, v in state.items() if k.startswith("graph_encoder.")}
        self.graph_encoder.load_state_dict(ge_state)

    def freeze_graph_encoder(self):
        for p in self.graph_encoder.parameters():
            p.requires_grad_(False)


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        act_layer: type[nn.Module] = nn.GELU,  # >>> MOD: GELU over SiLU.
        bias: bool = True,
        drop: float = 0.3,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features, bias=bias),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.net(x)
    