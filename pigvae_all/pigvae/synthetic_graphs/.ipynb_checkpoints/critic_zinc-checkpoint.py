from numpy import zeros
import torch
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss

def cross_entropy(trues_matrix, preds_matrix, device):

    weight_tensor = torch.tensor([0.2, 2.0, 1.0, 0.8, 1.0]).to(device)

    eps = 10e-12
    p = trues_matrix
    q = preds_matrix
    es = - torch.sum(p * torch.clamp_min(torch.log(q + eps), -100), dim = 0)
    e = torch.dot(es, weight_tensor)
    em = e / p.shape[0]

    return em

class Critic(torch.nn.Module):
    def __init__(self, hparams, device):
        super().__init__()
        self.alpha = hparams["kld_loss_scale"]
        self.beta = hparams["perm_loss_scale"]
        self.gamma = hparams["property_loss_scale"]
        self.vae = hparams["vae"]
        self.reconstruction_loss = GraphReconstructionLoss(device)
        self.perm_loss = PermutaionMatrixPenalty()
        self.property_loss = PropertyLoss()
        self.kld_loss = KLDLoss()

    def forward(self, true_nodes, true_edges, true_props, pred_nodes, pred_edges, pred_props, mask, perm, mu, logvar):
        
        #Loss 01 : Reconstruction
        recon_loss = self.reconstruction_loss(
            t_nodes = true_nodes,
            t_edges = true_edges,
            p_nodes = pred_nodes,
            p_edges = pred_edges,
            mask = mask
        )
        #Loss 02 : Length prediction and Permutation loss
        perm_loss = self.perm_loss(perm) #予測値
        property_loss = self.property_loss(
            input=true_props,
            target=pred_props
        )
        #Loss 03 : KL divergence
        
        if self.vae:
            kld_loss = self.kld_loss(mu, logvar)
            loss = {**recon_loss, "kld_loss": kld_loss, "perm_loss": perm_loss, "property_loss": property_loss}
        else:
            loss = {**recon_loss, "perm_loss": perm_loss, "property_loss": property_loss}
        
        return loss

class GraphReconstructionLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.edge_loss = CrossEntropyLoss()
        self.dist_loss = MSELoss()
        self.node_loss = BCEWithLogitsLoss()
        self.device = device

    def forward(self, t_nodes, t_edges, p_nodes, p_edges, mask):
        #mask = graph_true.mask
        sm = torch.nn.Softmax(dim = 1)

        adj_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
  
        edges_true = t_edges[adj_mask][:, 6:11]
        edges_pred = p_edges[adj_mask][:, 6:11]

        edges_pred_act = sm(edges_pred)

        edge_loss = cross_entropy(
            edges_true,
            edges_pred_act,
            self.device
        )
        
        node_loss = self.node_loss(
            input=p_nodes,
            target=t_nodes
        )
        loss = {
            "edge_loss": edge_loss,
            "node_loss": node_loss
        }
        return loss


class PropertyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = MSELoss()

    def forward(self, input, target):
        loss = self.mse_loss(
            input=input,
            target=target
        )
        return loss


class PermutaionMatrixPenalty(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def entropy(p, axis, normalize=True, eps=10e-12):
        if normalize:
            p = p / (p.sum(axis=axis, keepdim=True) + eps)
        e = - torch.sum(p * torch.clamp_min(torch.log(p), -100), axis=axis)
        return e

    def forward(self, perm, eps=10e-8):
        #print(perm.shape)
        perm = perm + eps
        entropy_col = self.entropy(perm, axis=1, normalize=False)
        entropy_row = self.entropy(perm, axis=2, normalize=False)
        loss = entropy_col.mean() + entropy_row.mean()
        return loss

class KLDLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1)
        loss = torch.mean(loss)
        return loss