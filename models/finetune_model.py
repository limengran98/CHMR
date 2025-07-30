import torch
import torch.nn as nn

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from .conv import GNN_node, GNN_node_Virtualnode


class FineTuneGNN(nn.Module):
    def __init__(
        self,
        args = None,
        num_tasks=None,
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        drop_ratio=0.5,
        graph_pooling="max",
        norm_layer="batch_norm",
        # return_tree=False
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
            self.graph2D_encoder = GNN_node_Virtualnode(
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
            self.graph2D_encoder = GNN_node(
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

        pro_dims=[1024+167, 300, 512, 1111+862, 1783+966, 978]


        self.modal_projectors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),        
                nn.Linear(dim, emb_dim),
                nn.SiLU()                
            )
            for dim in pro_dims
        ])

        self.dataset = args.dataset
        self.gamma = args.gamma
        self.dropout = args.task_dropout
        self.hidden = args.hidden

        self.task_decoder = MLP(1024+167+3*emb_dim, hidden_features= args.hidden * emb_dim, out_features=num_tasks, dropout = args.task_dropout)
        self.return_tree = False


    
    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)


        projected_features_1D = self.modal_projectors[0](batched_data.mol_features.float())
        projected_features_2D = self.modal_projectors[1](h_graph)
        projected_features_3D = self.modal_projectors[2](batched_data.unimol_features)
        
        if self.dataset in ['finetune-molhiv', 'finetune-molbace', 'finetune-molclintox', 'finetune-molsider']:
            task_out = self.gamma*self.task_decoder(torch.cat([batched_data.mol_features.float(), projected_features_1D, projected_features_2D, projected_features_3D], dim=1)) + batched_data.rf_pred.detach()
        else:
            task_out = self.task_decoder(torch.cat([batched_data.mol_features.float(), projected_features_1D, projected_features_2D, projected_features_3D], dim=1))
        if self.return_tree == True:
            return [batched_data.mol_features.float(),h_graph,batched_data.unimol_features,h_graph],[projected_features_1D, projected_features_2D, projected_features_3D]
        else:
            return task_out

    def load_pretrained_graph_encoder(self, model_path):

        saved_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        graph_encoder_state_dict = {key: value for key, value in saved_state_dict.items() if key.startswith('graph_encoder.')}
        graph_encoder_state_dict = {key.replace('graph_encoder.', ''): value for key, value in graph_encoder_state_dict.items()}
        self.graph_encoder.load_state_dict(graph_encoder_state_dict)


        modal_projectors_state_dict = {key: value for key, value in saved_state_dict.items() if key.startswith('modal_projectors.')}
        modal_projectors_state_dict = {key.replace('modal_projectors.', ''): value for key, value in modal_projectors_state_dict.items()}
        self.modal_projectors.load_state_dict(modal_projectors_state_dict)


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
        dropout=0.8,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.ln = nn.LayerNorm(hidden_features)  
        self.bn = nn.BatchNorm1d(hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x