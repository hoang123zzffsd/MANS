import torch
import torch.nn as nn
import torch.nn.functional as F
from .Model import Model


class MMRotatE(Model):

    def __init__(self, ent_tot, rel_tot, dim=100, img_emb=None, img_dim=4096,
                 margin=5.0, epsilon=2.0, test_mode='lp', beta=None):
        super(MMRotatE, self).__init__(ent_tot, rel_tot)


        self.dim_e = dim * 2
        self.dim_r = dim

                     
        self.margin = margin
        self.epsilon = epsilon
        self.img_dim = img_dim
        self.test_mode = test_mode


        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.img_proj = nn.Linear(self.img_dim, self.dim_e)
        # self.img_embeddings = nn.Embedding.from_pretrained(img_emb).requires_grad_(False)
	    self.img_embeddings = img_emb
        self.img_embeddings.requires_grad = False
        self.beta = beta

        
        self.ent_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), 
			requires_grad=False
		    )
        nn.init.uniform_(
			tensor = self.ent_embeddings.weight.data, 
			a=-self.ent_embedding_range.item(), 
			b=self.ent_embedding_range.item()
		)
        self.rel_embedding_range = nn.Parameter(
			torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), 
			requires_grad=False
		)
        nn.init.uniform_(
			tensor = self.rel_embeddings.weight.data, 
			a=-self.rel_embedding_range.item(), 
			b=self.rel_embedding_range.item()
		)


        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

    def _calc(self, h, t, r, mode):
        pi = self.pi_const
        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        
        re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

		
        
        if mode == "head_batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail
        
        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0).sum(dim = -1)
        return score.permute(1, 0).flatten()

    
    def forward(self, data, batch_size, neg_mode='normal', neg_num=1):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        if neg_mode == "adaptive":
            mode = data['mode']
            h_img_neg, t_img_neg = batch_h[batch_size:].detach(), batch_t[batch_size:].detach()
            r_neg = batch_r[batch_size:].detach()
            h_neg = self.ent_embeddings(h_img_neg)
            t_neg = self.ent_embeddings(t_img_neg)
            r_neg = self.rel_embeddings(r_neg)
            h_img_ent_emb = self.img_proj(self.img_embeddings(h_img_neg))
            t_img_ent_emb = self.img_proj(self.img_embeddings(t_img_neg))
            neg_score1 = self._calc(h_neg, t_neg, r_neg, mode) + self._calc(h_img_ent_emb, t_img_ent_emb, r_neg, mode)
            neg_score2 = (
                self._calc(h_img_ent_emb, t_neg, r_neg, mode)
                    + self._calc(h_neg, t_img_ent_emb, r_neg, mode)
            )
            selector = (neg_score2 < neg_score1).int()
            img_idx = torch.nonzero(selector).reshape((-1,))
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
        h_img_emb = self.img_proj(self.img_embeddings(h_img))
        t_img_emb = self.img_proj(self.img_embeddings(t_img))
        score = self.margin - (
                self._calc(h, t, r, mode)
                + self._calc(h_img_emb, t_img_emb, r, mode)
                + self._calc(h_img_emb, t, r, mode)
                + self._calc(h, t_img_emb, r, mode)
        )
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
        h_ent, h_img, t_ent, t_img = batch_h, batch_h, batch_t, batch_t
        mode = data['mode']
        h = self.ent_embeddings(h_ent)
        t = self.ent_embeddings(t_ent)
        r = self.rel_embeddings(batch_r)
        h_img_emb = self.img_proj(self.img_embeddings(h_img))
        t_img_emb = self.img_proj(self.img_embeddings(t_img))
        score = self.margin - self._calc(h ,t_img_emb, r, mode)
        return score
 
    def predict(self, data):
        if self.test_mode == 'cmlp':
            score = -self.cross_modal_score_ent2img(data)
        else:
            score = -self.forward(data, batch_size=None, neg_mode='normal')

        return score.cpu().data.numpy()



    def set_test_mode(self, new_mode):
        self.test_mode = new_mode

    def get_rel_rank(self, data):
        head, tail, rel = data
        h_img_emb = self.img_proj(self.img_embeddings(head))
        t_img_emb = self.img_proj(self.img_embeddings(tail))
        relations = self.rel_embeddings.weight
        h = h_img_emb.reshape(-1, h_img_emb.shape[0]).expand((relations.shape[0], h_img_emb.shape[0]))
        t = t_img_emb.reshape(-1, t_img_emb.shape[0]).expand((relations.shape[0], t_img_emb.shape[0]))
        scores = self._calc(h, t, relations, mode='normal')
        ranks = torch.argsort(scores)
        rank = 0
        for (index, val) in enumerate(ranks):
            if val.item() == rel.item():
                rank = index
                break
        return rank + 1

    def get_entity_embs(self, index):
        index = torch.LongTensor(index)
        structural_embs = self.ent_embeddings(index)
        visual_embs = self.img_proj(self.img_embeddings(index))
        return structural_embs, visual_embs
