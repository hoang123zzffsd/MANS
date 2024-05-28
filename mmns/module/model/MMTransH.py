import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .Model import Model


class MMTransH(Model):

    def __init__(self, ent_tot, rel_tot, dim=100, p_norm=1, img_emb=None,
                 img_dim=4096, norm_flag=True, margin=None, epsilon=None,
                 test_mode='lp', beta=None):
        super(MMTransH, self).__init__(ent_tot, rel_tot)

        self.dim = dim
        self.margin = margin
        self.epsilon = epsilon
        self.norm_flag = norm_flag
        self.p_norm = p_norm
        self.img_dim = img_dim
        self.test_mode = test_mode

        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)
        self.norm_vector = nn.Embedding(self.rel_tot, self.dim)  # norm vector for each relation
        self.img_proj = nn.Linear(self.img_dim, self.dim)
        self.img_embeddings = img_emb
        self.img_embeddings.requires_grad = False
        self.beta = beta
        if margin is None or epsilon is None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_uniform_(self.norm_vector.weight.data)
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
                tensor=self.norm_vector.weight.data,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if margin is not None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _project(self, emb, norm):
        norm = F.normalize(norm, p=2, dim=-1)
        return emb - torch.sum(emb * norm, dim=-1, keepdim=True) * norm

    def _calc(self, h, t, r, mode):
        h = self._project(h, norm)
        t = self._project(t, norm)
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        if mode != 'normal':
            h = h.view(-1, r.shape[0], h.shape[-1])
            t = t.view(-1, r.shape[0], t.shape[-1])
            r = r.view(-1, r.shape[0], r.shape[-1])
        if mode == 'head_batch':
            score = h + (r - t)
        else:
            score = (h + r) - t
        score = torch.norm(score, self.p_norm, -1).flatten()
        return score

    def forward(self, data, batch_size, neg_mode='normal', neg_num=1):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        norm = self.norm_vector(batch_r)
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        h = self._project(h, norm)
        t = self._project(t, norm)
        r = self.rel_embeddings(batch_r)
        
        if neg_mode == "adaptive":
            mode = data['mode']
            h_img_neg, t_img_neg = batch_h[batch_size:].detach(), batch_t[batch_size:].detach()
            r_neg = batch_r[batch_size:].detach()
            h_neg = self.ent_embeddings(h_img_neg)
            t_neg = self.ent_embeddings(t_img_neg)
            h_neg = self._project(h_neg, self.norm_vector(r_neg))
            t_neg = self._project(t_neg, self.norm_vector(r_neg))
            r_neg = self.rel_embeddings(r_neg)
            h_img_ent_emb = self.img_proj(self.img_embeddings(h_img_neg))
            t_img_ent_emb = self.img_proj(self.img_embeddings(t_img_neg))
            h_img_ent_emb = self._project(h_img_ent_emb, self.norm_vector(r_neg))
            t_img_ent_emb = self._project(t_img_ent_emb, self.norm_vector(r_neg))
            neg_score1 = self._calc(h_neg, t_neg, r_neg, mode) + self._calc(h_img_ent_emb, t_img_ent_emb, r_neg, mode)
            neg_score2 = (
                self._calc(h_img_ent_emb, t_neg, r_neg, mode)
                    + self._calc(h_neg, t_img_ent_emb, r_neg, mode)
            )
            selector = (neg_score2 < neg_score1).int()
            img_idx = torch.nonzero(selector).reshape((-1, ))
            p = img_idx.shape[0] / (batch_size * neg_num)
            num = int(neg_num * p * batch_size)
            h_ent, h_img, t_ent, t_img = batch_h.clone(), batch_h.clone(), batch_t.clone(), batch_t.clone()
            h_ent[batch_size: batch_size + num] = batch_h[0: num].clone()
            t_ent[batch_size: batch_size + num] = batch_t[0: num].clone()
        else:
            num = int(neg_num * self.beta * batch_size) if batch_size != None else 0
            h_ent, h_img, t_ent, t_img = None, None, None, None
            
            if neg_mode == "normal":
                h_ent, h_img, t_ent, t_img = batch_h, batch_h, batch_t, batch_t
            else:
                if neg_mode == "img":
                    h_img, t_img = batch_h, batch_t
                    h_ent = torch.tensor(batch_h[:batch_size]).repeat(neg_num + 1)
                    t_ent = torch.tensor(batch_t[:batch_size]).repeat(neg_num + 1)
                elif neg_mode == "hybrid":
                    h_ent, h_img, t_ent, t_img = batch_h.clone(), batch_h.clone(), batch_t.clone(), batch_t.clone()
                    h_ent[batch_size: batch_size + num] = batch_h[0: num].clone()
                    t_ent[batch_size: batch_size + num] = batch_t[0: num].clone()
        
        mode = data['mode']
        h = self.ent_embeddings(h_ent)
        t = self.ent_embeddings(t_ent)
        r = self.rel_embeddings(batch_r)
        h = self._project(h, self.norm_vector(batch_r))
        t = self._project(t, self.norm_vector(batch_r))
        h_img_emb = self.img_proj(self.img_embeddings(h_img))
        t_img_emb = self.img_proj(self.img_embeddings(t_img))
        h_img_emb = self._project(h_img_emb, self.norm_vector(batch_r))
        t_img_emb = self._project(t_img_emb, self.norm_vector(batch_r))
        
        score = (
                self._calc(h, t, r, mode)
                + self._calc(h_img_emb, t_img_emb, r, mode)
                + self._calc(h_img_emb, t, r, mode)
                + self._calc(h, t_img_emb, r, mode)
        )
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
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul

    def cross_modal_score_ent2img(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        norm = self.norm_vector(batch_r)
        h_ent, h_img, t_ent, t_img = batch_h, batch_h, batch_t, batch_t
        mode = data['mode']
        h = self.ent_embeddings(h_ent)