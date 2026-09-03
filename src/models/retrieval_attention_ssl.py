"""Retrieval attention SSL over labeled+unlabeled train memory only."""
from __future__ import annotations
from typing import Any
import numpy as np
from src.exceptions import OptionalDependencyError
from src.models.torch_utils import normalize_probability_matrix, validate_probability_matrix

def _require_torch():
    try:
        import torch, torch.nn as nn, torch.nn.functional as F
    except ImportError as e:
        raise OptionalDependencyError("torch","needed for retrieval_attention_ssl") from e
    return torch, nn, F

class RetrievalAttentionSSL:
    name="retrieval_attention_ssl"
    def __init__(self, random_state=0, n_classes=2, hidden_dim=64, embedding_dim=32,
                 k=8, max_epochs=30, patience=5, lr=1e-3, device="cpu",
                 memory_mode="labeled_plus_unlabeled", conf_threshold=0.9, **kwargs):
        self.random_state=int(random_state); self.n_classes=int(n_classes)
        self.cfg=dict(hidden_dim=hidden_dim,embedding_dim=embedding_dim,k=k,max_epochs=max_epochs,
                      patience=patience,lr=lr,device=device,memory_mode=memory_mode,
                      conf_threshold=conf_threshold,**kwargs)
        self.classes_=None; self.training_meta={}; self._net=None; self._device=None
        self._mem_X=None; self._mem_y=None; self._mem_type=None; self._mem_reliable=None

    def fit(self, X_labeled, y_labeled, X_unlabeled=None, X_val=None, y_val=None):
        torch,nn,F=_require_torch(); torch.manual_seed(self.random_state); np.random.seed(self.random_state)
        X_l=np.asarray(X_labeled,np.float32); y_l=np.asarray(y_labeled)
        self.classes_=np.unique(y_l); c2i={c:i for i,c in enumerate(self.classes_)}
        y_loc=np.array([c2i[c] for c in y_l],np.int64)
        X_u=np.asarray(X_unlabeled,np.float32) if X_unlabeled is not None and len(X_unlabeled) else np.empty((0,X_l.shape[1]),np.float32)
        device=torch.device("cpu" if self.cfg["device"] in (None,"auto","cpu") else self.cfg["device"])
        if str(device).startswith("cuda") and not torch.cuda.is_available(): device=torch.device("cpu")
        self._device=device
        # Build memory: train only
        if self.cfg["memory_mode"]=="labeled_only" or len(X_u)==0:
            mem_X=X_l; mem_y=y_loc; mem_type=np.zeros(len(X_l),np.int64); mem_rel=np.ones(len(X_l),bool)
        else:
            # temporary teacher for reliable PL tokens
            from sklearn.linear_model import LogisticRegression
            clf=LogisticRegression(max_iter=1000,random_state=self.random_state).fit(X_l,y_loc)
            pu=clf.predict_proba(X_u); pred=pu.argmax(1); conf=pu.max(1)
            rel=conf>=float(self.cfg["conf_threshold"])
            mem_X=np.vstack([X_l,X_u]); mem_y=np.concatenate([y_loc,pred]); 
            mem_type=np.concatenate([np.zeros(len(X_l),np.int64), np.ones(len(X_u),np.int64)])
            mem_rel=np.concatenate([np.ones(len(X_l),bool), rel])
        self._mem_X=mem_X; self._mem_y=mem_y; self._mem_type=mem_type; self._mem_reliable=mem_rel
        d_in=X_l.shape[1]; C=len(self.classes_); h=self.cfg["hidden_dim"]; e=self.cfg["embedding_dim"]; k=min(int(self.cfg["k"]), max(1,len(mem_X)-1))
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc=nn.Sequential(nn.Linear(d_in,h),nn.ReLU(),nn.Linear(h,e))
                self.class_emb=nn.Embedding(C,e); self.type_emb=nn.Embedding(2,e)
                self.attn=nn.MultiheadAttention(e, num_heads=4, batch_first=True)
                self.gate=nn.Linear(2*e,e); self.cls=nn.Linear(e,C)
                self.ema_enc=nn.Sequential(nn.Linear(d_in,h),nn.ReLU(),nn.Linear(h,e))
                for p in self.ema_enc.parameters(): p.requires_grad=False
            def encode(self,x,ema=False):
                z=self.ema_enc(x) if ema else self.enc(x); return F.normalize(z,dim=-1)
            def forward(self, x, mem_x, mem_y, mem_type, mem_rel, self_idx=None):
                q=self.encode(x)  # B,e
                with torch.no_grad():
                    m=self.encode(mem_x, ema=True)
                # retrieve top-k excluding self
                sim=q @ m.T  # B,N
                if self_idx is not None:
                    for i,si in enumerate(self_idx):
                        if si is not None and 0<=si<sim.shape[1]: sim[i,si]=-1e9
                topv, topi=torch.topk(sim, k=min(k, sim.shape[1]), dim=1)
                B=x.size(0)
                neigh=m[topi]  # B,k,e
                # memory tokens
                lab_tok=self.class_emb(mem_y.clamp(0,C-1))[topi]
                typ_tok=self.type_emb(mem_type[topi])
                # zero class token for unreliable unlabeled
                rel_mask=mem_rel[topi].float().unsqueeze(-1)
                tokens=neigh + typ_tok + lab_tok*rel_mask
                # also attend to class prototypes
                proto=self.class_emb.weight.unsqueeze(0).expand(B,-1,-1)
                memory=torch.cat([tokens, proto], dim=1)
                attn_out,_=self.attn(q.unsqueeze(1), memory, memory)
                attn_out=attn_out.squeeze(1)
                g=torch.sigmoid(self.gate(torch.cat([q,attn_out],-1)))
                comb=g*q+(1-g)*attn_out
                return self.cls(comb), q, topi, topv
            @torch.no_grad()
            def update_ema(self, decay=0.99):
                for p,ep in zip(self.enc.parameters(), self.ema_enc.parameters()):
                    ep.data.mul_(decay).add_(p.data, alpha=1-decay)
        net=Net().to(device); 
        # init ema = enc
        net.ema_enc.load_state_dict(net.enc.state_dict())
        opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=self.cfg["lr"])
        Xl=torch.tensor(X_l,device=device); yl=torch.tensor(y_loc,device=device)
        Mx=torch.tensor(mem_X,device=device); My=torch.tensor(mem_y,device=device)
        Mt=torch.tensor(mem_type,device=device); Mr=torch.tensor(mem_rel,device=device)
        best=None; best_loss=1e9; wait=0; best_epoch=0
        n=len(X_l)
        for epoch in range(int(self.cfg["max_epochs"])):
            net.train(); idx=np.random.RandomState(self.random_state+epoch).permutation(n)
            total=0.0; steps=0
            bs=min(64,n)
            for s in range(0,n,bs):
                batch=idx[s:s+bs]
                xb=Xl[batch]; yb=yl[batch]
                # self indices in memory for labeled rows = batch positions in labeled prefix
                self_idx=batch.tolist()
                logits,_,_,_=net(xb,Mx,My,Mt,Mr,self_idx=self_idx)
                loss=nn.functional.cross_entropy(logits,yb)
                opt.zero_grad(); loss.backward(); opt.step(); net.update_ema()
                total+=float(loss.detach().cpu()); steps+=1
            score=total/max(steps,1)
            if score+1e-6<best_loss:
                best_loss=score; best={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}; best_epoch=epoch; wait=0
            else:
                wait+=1
                if wait>=int(self.cfg["patience"]): break
        if best: net.load_state_dict(best)
        self._net=net
        # attention mass diagnostics
        net.eval()
        with torch.no_grad():
            _,_,topi,topv=net(Xl[:min(128,len(Xl))],Mx,My,Mt,Mr,self_idx=list(range(min(128,len(Xl)))))
            types=Mt[topi].cpu().numpy(); lab_mass=float((types==0).mean()); u_mass=float((types==1).mean())
        self.training_meta={
            "method_fidelity":"novel_experimental","protocol":"inductive","uses_unlabeled_data": self.cfg["memory_mode"]!="labeled_only",
            "attention_k":k,"memory_size":len(mem_X),"memory_mode":self.cfg["memory_mode"],
            "labeled_attention_mass":lab_mass,"unlabeled_attention_mass":u_mass,
            "best_epoch":best_epoch,"best_val_loss":best_loss,
            "memory_contains_val_or_test":False,
        }
        return self

    def predict_proba(self,X):
        torch,nn,F=_require_torch(); self._net.eval()
        X=np.asarray(X,np.float32)
        Mx=torch.tensor(self._mem_X,device=self._device); My=torch.tensor(self._mem_y,device=self._device)
        Mt=torch.tensor(self._mem_type,device=self._device); Mr=torch.tensor(self._mem_reliable,device=self._device)
        outs=[]
        with torch.no_grad():
            for s in range(0,len(X),256):
                xb=torch.tensor(X[s:s+256],device=self._device)
                logits,_,_,_=self._net(xb,Mx,My,Mt,Mr,self_idx=None)
                outs.append(F.softmax(logits,1).cpu().numpy())
        p=normalize_probability_matrix(np.vstack(outs))
        p=validate_probability_matrix(p, n_classes=len(self.classes_))
        out=np.zeros((len(X), int(self.classes_.max())+1), np.float64)
        for j,c in enumerate(self.classes_): out[:,int(c)]=p[:,j]
        return normalize_probability_matrix(out)
    def predict(self,X): return self.predict_proba(X).argmax(1)

def build_retrieval_model(random_state=0,n_classes=2,method_name="unlabeled_attention_ssl",**kw):
    model = RetrievalAttentionSSL(random_state=random_state,n_classes=n_classes,**kw)
    model.name = method_name
    return model
