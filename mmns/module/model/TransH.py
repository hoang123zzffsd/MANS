import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model

class TransH(Model):

    def __init__(self, ent_tot, rel_tot, dim=100, p_norm=1, norm_flag=True, margin=None, epsilon=None):
        super(TransH, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.norm_embeddings = nn.Embedding(self.rel_tot, self.dim)  # Vector pháp tuyến cho mỗi quan hệ

        if margin is None or epsilon is None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_uniform_(self.norm_embeddings.weight.data)
        else:
            self.embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.norm_embeddings.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if margin is not None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def project_to_hyperplane(self, entity, norm):
        return entity - torch.sum(entity * norm, dim=1, keepdim=True) * norm

    def _calc(self, h, t, r, norm, mode):
        h_proj = self.project_to_hyperplane(h, norm)
        t_proj = self.project_to_hyperplane(t, norm)
        
        if self.norm_flag:
            h_proj = F.normalize(h_proj, 2, -1)
            r = F.normalize(r, 2, -1)
            t_proj = F.normalize(t_proj, 2, -1)

        if mode != 'normal':
            h_proj = h_proj.view(-1, r.shape[0], h_proj.shape[-1])
            t_proj = t_proj.view(-1, r.shape[0], t_proj.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])

        if mode == 'head_batch':
            score = h_proj + (r - t_proj)
        else:
            score = (h_proj + r) - t_proj

        score = torch.norm(score, self.p_norm, -1).flatten()
        return score

    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        mode = data['mode']

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        norm = self.norm_embeddings(batch_r)

        score = self._calc(h, t, r, norm, mode)
        
        if self.margin_flag:
            return self.margin - score
        else:
            return score

    def regularization(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']

        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        norm = self.norm_embeddings(batch_r)

        regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2) + torch.mean(norm ** 2)) / 4
        return regul

    def predict(self, data):
        score = self.forward(data)
        if self.margin_flag:
            score = self.margin - score
            return score.cpu().data.numpy()
        else:
            return score.cpu().data.numpy()
