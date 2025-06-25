import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_cluster import random_walk

from .misc import AverageMeter


def pretrain_loss_ce(pred, target, weight, nan_mask):
    expanded_weight = weight.unsqueeze(2).expand_as(target)
    loss = F.binary_cross_entropy_with_logits(
        pred[~nan_mask],
        target[~nan_mask],
        weight=expanded_weight[~nan_mask],
        reduction="none",
    )
    return loss.mean()

def pretrain_func(
    args, model, train_loaders, context_graph, optimizer, scheduler, epoch
):
    
    criterion = pretrain_loss_ce
    if not args.no_print:
        p_bar = tqdm(range(args.steps))
    batch_time = AverageMeter()
    (losses_tot, losses_prior, losses_mol, losses_gene, losses_cell, losses_exp) = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    device = args.device
    model.train()

    context_node_type = context_graph.type
    mol_target = context_graph.mol_target
    gene_target = context_graph.gene_target
    cell_target = context_graph.cell_target
    express_target = context_graph.express_target
    edge_weight = context_graph.weight
    edge_weight = torch.cat(
        [
            edge_weight,
            torch.zeros(1, dtype=edge_weight.dtype, device=edge_weight.device),
        ]
    )
    index_map = {value: index for index, value in enumerate(context_node_type)}

    for batch_idx in range(args.steps):
        end = time.time()
        model.zero_grad()
        try:
            batched_data = next(train_loaders["train_iter"])
        except:
            train_loaders["train_iter"] = iter(train_loaders["train_loader"])
            batched_data = next(train_loaders["train_iter"])
        batched_data = batched_data.to(device)

        start_indices = [index_map[value] for value in batched_data.type]
        start_indices = torch.tensor(start_indices).long()
        start_indices = start_indices.view(-1, 1).repeat(1, 1).view(-1)
        batched_walk, batched_edge_seq = random_walk(
            context_graph.edge_index[0],
            context_graph.edge_index[1],
            start_indices,
            args.walk_length,
            num_nodes=context_graph.num_nodes,
            return_edge_indices=True,
        )

        # batched_walk = batched_walk[:, 1:]
        batched_path_weight = (
            edge_weight[batched_edge_seq]
            .view(-1, args.walk_length)
            .cumprod(dim=-1)
            .to(device)
        )
        ## if count starting
        batched_path_weight = torch.cat(
            [
                torch.ones(
                    batched_path_weight.size(0), 1, device=batched_path_weight.device
                ),
                batched_path_weight,
            ],
            dim=1,
        )
        
        batched_mol_target = mol_target[batched_walk].to(device)
        batched_gene_target = gene_target[batched_walk].to(device)
        batched_cell_target = cell_target[batched_walk].to(device)
        batched_express_target = express_target[batched_walk].to(device)

        nan_mask_mol = torch.isnan(batched_mol_target)
        nan_mask_gene = torch.isnan(batched_gene_target)
        nan_mask_cell = torch.isnan(batched_cell_target)
        nan_mask_express = torch.isnan(batched_express_target)

        if batched_data.x.shape[0] == 1 or batched_data.batch[-1] == 0:
            continue
        else:
            pred_list, loss_rout_graph = model(batched_data)
            pred_mol, pred_gene, pred_cell, pred_express = pred_list
            #pred_mol = pred_list


            pred_mol = pred_mol.unsqueeze(1).expand(-1, batched_mol_target.size(1), -1)
            pred_gene = pred_gene.unsqueeze(1).expand(
                -1, batched_gene_target.size(1), -1
            )
            pred_cell = pred_cell.unsqueeze(1).expand(
                -1, batched_cell_target.size(1), -1
            )
            pred_express = pred_express.unsqueeze(1).expand(
                -1, batched_express_target.size(1), -1
            )

            #loss_prior = prior_criterion(args, pz_given_x)
            
            loss_mol = criterion(
                pred_mol, batched_mol_target, batched_path_weight, nan_mask_mol
            )
            loss_gene = criterion(
                pred_gene, batched_gene_target, batched_path_weight, nan_mask_gene
            )
            loss_cell = criterion(
                pred_cell, batched_cell_target, batched_path_weight, nan_mask_cell
            )
            loss_exp = criterion(
                pred_express,
                batched_express_target,
                batched_path_weight,
                nan_mask_express,
            )
            loss = args.prior * loss_rout_graph + loss_mol + loss_gene + loss_cell #+ loss_exp #+ args.prior * loss_rout_graph

        loss.backward()
        optimizer.step()
        scheduler.step()
        losses_tot.update(loss.item())
        losses_prior.update(args.prior * loss_rout_graph.item())
        losses_mol.update(loss_mol.item())
        losses_gene.update(loss_gene.item())
        losses_cell.update(loss_cell.item())
        losses_exp.update(loss_exp.item())

        batch_time.update(time.time() - end)
        end = time.time()
        if not args.no_print:
            log_message = "Train Epoch: {epoch}/{epochs:3}. Iter: {batch:2}/{iter:2}. LR: {lr:1}e-4. Batch: {bt:.1f}s. Loss (Total): {loss_tot:.2f}. prior: {loss_prior:.4f}. mol: {loss_mol:.4f}. gene: {loss_gene:.4f}. cell: {loss_cell:.4f}. express: {loss_exp:.4f}.".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.steps,
                lr=args.lr * 10000,
                bt=batch_time.avg,
                loss_tot=losses_tot.avg,
                loss_prior=losses_prior.avg,
                loss_mol=losses_mol.avg,
                loss_gene=losses_gene.avg,
                loss_cell=losses_cell.avg,
                loss_exp=losses_exp.avg,
            )
            p_bar.set_description(log_message)
            p_bar.update()
    if not args.no_print:
        p_bar.close()
        args.logger.info(log_message)

    return losses_tot.avg, train_loaders


