"""Prototype-alignment SSL (processed features)."""
from __future__ import annotations
from typing import Any
import numpy as np
from src.exceptions import OptionalDependencyError
from src.models.torch_utils import normalize_probability_matrix, validate_probability_matrix

def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as e:
        raise OptionalDependencyError("torch", "needed for prototype_alignment_ssl") from e
    return torch, nn, F

class PrototypeAlignmentSSL:
    name = "prototype_alignment_ssl"
    def __init__(self, random_state=0, n_classes=2, hidden_dim=64, embedding_dim=32,
                 max_epochs=40, patience=6, batch_size=128, lr=1e-3,
                 lambda_pl=0.5, lambda_proto=0.5, lambda_margin=0.2, lambda_consistency=0.1,
                 conf_threshold=0.9, device="cpu", **kwargs):
        self.random_state=int(random_state); self.n_classes=int(n_classes)
        self.cfg=dict(hidden_dim=hidden_dim, embedding_dim=embedding_dim, max_epochs=max_epochs,
                      patience=patience, batch_size=batch_size, lr=lr, lambda_pl=lambda_pl,
                      lambda_proto=lambda_proto, lambda_margin=lambda_margin,
                      lambda_consistency=lambda_consistency, conf_threshold=conf_threshold,
                      device=device, **kwargs)
        self.classes_=None; self.training_meta={}; self._net=None; self._device=None

    def fit(self, X_labeled, y_labeled, X_unlabeled=None, X_val=None, y_val=None):
        torch, nn, F = _require_torch()
        torch.manual_seed(self.random_state); np.random.seed(self.random_state)
        X_l=np.asarray(X_labeled,np.float32); y_l=np.asarray(y_labeled)
        self.classes_=np.unique(y_l); class_to= {c:i for i,c in enumerate(self.classes_)}
        y_loc=np.array([class_to[c] for c in y_l], dtype=np.int64)
        X_u=np.asarray(X_unlabeled,np.float32) if X_unlabeled is not None and len(X_unlabeled) else np.empty((0,X_l.shape[1]),np.float32)
        device=torch.device(self.cfg["device"] if self.cfg["device"]!="auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._device=device
        d_in=X_l.shape[1]; C=len(self.classes_); h=self.cfg["hidden_dim"]; e=self.cfg["embedding_dim"]
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc=nn.Sequential(nn.Linear(d_in,h),nn.ReLU(),nn.Dropout(0.1),nn.Linear(h,e))
                self.cls=nn.Linear(e,C)
            def forward(self,x):
                z=self.enc(x); return self.cls(z), F.normalize(z,dim=-1)
        net=Net().to(device); opt=torch.optim.Adam(net.parameters(), lr=self.cfg["lr"], weight_decay=1e-4)
        Xl=torch.tensor(X_l,device=device); yl=torch.tensor(y_loc,device=device)
        Xu=torch.tensor(X_u,device=device) if len(X_u) else None
        best_state=None; best_val=float("inf"); wait=0; best_epoch=0
        has_val = X_val is not None and y_val is not None and len(X_val)>0
        if has_val:
            Xv=torch.tensor(np.asarray(X_val,np.float32),device=device)
            yv=torch.tensor(np.array([class_to.get(c,-1) for c in np.asarray(y_val)]),device=device)
            mask=yv>=0; Xv=Xv[mask]; yv=yv[mask]
        margin=1.0
        for epoch in range(int(self.cfg["max_epochs"])):
            net.train(); opt.zero_grad()
            logits,z=net(Xl); loss_sup=F.cross_entropy(logits,yl)
            loss=loss_sup
            # prototypes from labeled
            protos=[]
            for c in range(C):
                m=(yl==c); protos.append(z[m].mean(0) if m.any() else torch.zeros(e,device=device))
            P=torch.stack(protos,0)
            # class separation
            if C>1:
                dist=((P[:,None,:]-P[None,:,:]).pow(2).sum(-1)+1e-8).sqrt()
                eye=torch.eye(C,device=device).bool()
                sep=F.relu(margin-dist.masked_fill(eye, margin+1)).mean()
                loss=loss+self.cfg["lambda_margin"]*sep
            n_rel=0
            if Xu is not None and len(Xu)>0:
                with torch.no_grad():
                    log_u,_=net(Xu); conf=F.softmax(log_u,1).max(1).values; pred=log_u.argmax(1)
                rel=conf>=float(self.cfg["conf_threshold"]); n_rel=int(rel.sum().item())
                if rel.any():
                    log_u2,z_u=net(Xu[rel]); pred_r=pred[rel]
                    loss=loss+self.cfg["lambda_pl"]*F.cross_entropy(log_u2,pred_r)
                    # attract to predicted prototype
                    target=P[pred_r]
                    loss=loss+self.cfg["lambda_proto"]*((z_u-target).pow(2).sum(-1).mean())
                    # consistency: detach teacher
                    with torch.no_grad():
                        _,z_t=net(Xu[rel])
                    loss=loss+self.cfg["lambda_consistency"]*((z_u-z_t).pow(2).mean())
            loss.backward(); opt.step()
            # val
            score=float(loss.detach().cpu())
            if has_val and len(yv)>0:
                net.eval()
                with torch.no_grad():
                    lv,_=net(Xv); score=float(F.cross_entropy(lv,yv).cpu())
            if score+1e-6<best_val:
                best_val=score; best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}; best_epoch=epoch; wait=0
            else:
                wait+=1
                if wait>=int(self.cfg["patience"]): break
        if best_state: net.load_state_dict(best_state)
        self._net=net
        # diagnostics
        net.eval()
        with torch.no_grad():
            _,z_all=net(Xl); P=torch.stack([z_all[yl==c].mean(0) if (yl==c).any() else torch.zeros(e,device=device) for c in range(C)])
            intra=[]; 
            for c in range(C):
                m=(yl==c)
                if m.any(): intra.append(float(((z_all[m]-P[c]).pow(2).sum(-1).mean()).cpu()))
            inter=[]
            for i in range(C):
                for j in range(i+1,C):
                    inter.append(float(((P[i]-P[j]).pow(2).sum()).sqrt().cpu()))
            var=float(z_all.var(0).mean().cpu())
        self.training_meta={
            "method_fidelity":"novel_experimental","protocol":"inductive","uses_unlabeled_data":True,
            "best_epoch":best_epoch,"best_val_loss":best_val,
            "intra_class_distances":intra,"inter_prototype_distances":inter,
            "embedding_variance":var,"representation_collapse_suspect":bool(var<1e-6),
            "n_reliable_unlabeled_last":n_rel if 'n_rel' in dir() else 0,
            "loss_weights":{k:self.cfg[k] for k in ("lambda_pl","lambda_proto","lambda_margin","lambda_consistency")},
        }
        return self

    def predict_proba(self,X):
        torch,nn,F=_require_torch(); self._net.eval()
        with torch.no_grad():
            logits,_=self._net(torch.tensor(np.asarray(X,np.float32),device=self._device))
            p=normalize_probability_matrix(F.softmax(logits,1).cpu().numpy())
            p=validate_probability_matrix(p, n_classes=len(self.classes_))
        # map local to global class ids
        out=np.zeros((len(X), int(self.classes_.max())+1), dtype=np.float64)
        for j,c in enumerate(self.classes_): out[:,int(c)]=p[:,j]
        return normalize_probability_matrix(out)
    def predict(self,X): return self.predict_proba(X).argmax(1)

def build_prototype_model(random_state=0,n_classes=2,method_name="embedding_alignment_ssl",**kw):
    model = PrototypeAlignmentSSL(random_state=random_state,n_classes=n_classes,**kw)
    model.name = method_name
    return model
