"""All 42 ranked method families, with bounded, declared tuning grids."""
import copy
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .cache import validate
from .data import seed_all, assert_disjoint
from .gates import Gate, features
from .losses import objective, signed_scores
from .metrics import metrics

IDS = ['U0','S1','S2','H1','H2','H3','S3','D1','D2','D3','D4','S4','H4','R1','R2','R3','R4',
       'S5','H5','R5','R6','D5','D6','D7','D8','D9','D10','S6','D11','D12','D13','D14','D15',
       'D16','D17','D18','H6','R7','D19','S7','R8','X0']
NEURAL = {'D1','D2','D3','D4','D11','D12','D13','D14','D15','D16','D17','D18','D19'}


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def normalize(w):
    w = np.asarray(w, dtype=float)
    w = np.maximum(w, 0)
    total = w.sum(-1, keepdims=True)
    return np.divide(w, total, out=np.full_like(w, 1 / w.shape[-1]), where=total > 0)


def oracle_coverage(p, y):
    """Diagnostic only: no deployable predictor is constructed."""
    return float((p.argmax(-1) == np.asarray(y)[:, None]).any(1).mean())


def candidates(method):
    if method in NEURAL:
        return [dict(kl=k, lr=.001) for k in (0., .01, .1)]
    if method in ('S1', 'S3', 'S6'):
        return [dict(kl=k) for k in (0., .01, .1)]
    if method in ('H1','H2','H3','H4'):
        return [dict(beta=b, macro=macro) for macro in ([False, True] if method in ('H2','H4') else [False])
                for b in (.5, 1., 2., 5., 10.)]
    if method == 'H6':
        return [dict(beta=b, rho=r) for b in (1., 5.) for r in (.1, 1.)]
    if method in ('R2','R6'):
        return [dict(k=k) for k in (3,5,7,10)]
    if method == 'R3':
        return [dict(k=k, beta=b) for k in (3,5,7,10) for b in (1.,5.)]
    if method == 'R4':
        return [dict(variant=v) for v in ('vote','median')]
    if method == 'S5':
        return [dict(rho=r) for r in (.01,.1,1.)]
    if method in ('D5','D6','D7'):
        return [dict(depth=d) for d in ([2,3,4] if method == 'D5' else [3,5])]
    if method in ('D8','S7'):
        return [dict(C=c) for c in (.1,1.,10.)]
    if method == 'D9':
        return [dict(k=k) for k in (15,50)]
    return [{}]


class Predictor:
    def __init__(self, method, cfg=None):
        if method == 'X0':
            raise ValueError('Oracle is diagnostic only and prohibited at inference')
        self.method, self.cfg = method, dict(cfg or {})
        self.state, self.model = {}, None
        self.threshold = .5
        self.params = 0
        self.train_seconds = 0.

    def calibrated(self, data):
        if 'temperature' not in self.state:
            return data
        p = data['probs']
        p = softmax(np.log(p.clip(1e-9)) / self.state['temperature'][None, :, None])
        return data | dict(probs=p)

    def feature_data(self, data):
        cfg = self.cfg
        f, _ = features(data, context=self.method == 'D4', distance=self.method == 'D3',
                        embedding=self.method == 'D12', remove=cfg.get('remove', ()),
                        projection=self.state.get('projection'))
        if 'mean' in self.state:
            f = (f - self.state['mean']) / self.state['std']
        return f

    def fit(self, train, select, seed, *, epochs=100, patience=10, progress_path=None):
        """Only meta-training and meta-selection data accepted; neither has groups."""
        if 'group' in train or 'group' in select:
            raise ValueError('Group annotations must remain outside model fitting')
        assert_disjoint(train['sample_id'],select['sample_id'])
        seed_all(seed)
        start = time.perf_counter()
        p, y = validate(train['probs']), np.asarray(train['y'])
        n, m, c = p.shape
        self.members = m
        self.classes = c
        true = p[np.arange(n)[:, None], np.arange(m)[None, :], y[:, None]]
        correct = p.argmax(-1) == y[:, None]
        cfg, method = self.cfg, self.method
        if cfg.get('calibrate') or method == 'D19':
            temps = np.geomspace(.25,4.,25)
            scores = np.stack([-np.log(softmax(np.log(p.clip(1e-9))/t)[np.arange(n)[:,None],np.arange(m),y[:,None]].clip(1e-9)).mean(0) for t in temps])
            self.state['temperature'] = temps[scores.argmin(0)]
            p = self.calibrated(train)['probs']
            train, select = self.calibrated(train), self.calibrated(select)
        if method in NEURAL:
            self._fit_neural(train, select, seed, epochs, patience, progress_path)
        elif method in ('S1','S3','S4','S5','S6'):
            self.state['weights'] = self._static(p, y, method, cfg, epochs=max(100, epochs))
            self.params = self.state['weights'].size
            if method == 'S4' and c == 2:
                inner = np.zeros((n, c))
                for a,b in StratifiedKFold(3, shuffle=True, random_state=seed).split(p, y):
                    w = self._static(p[a],y[a],'S1',{},100)
                    inner[b] = np.einsum('nmc,m->nc',p[b],w)
                thresholds = np.linspace(.3,.7,41)
                self.threshold = float(max(thresholds,key=lambda t: (metrics(inner,y,threshold=t)['accuracy'],-abs(t-.5))))
        elif method == 'S2':
            from scipy.optimize import minimize
            pt = p[np.arange(n)[:,None],np.arange(m),y[:,None]]
            fun = lambda w: -np.log((pt@w).clip(1e-9)).mean()
            jac = lambda w: -(pt/(pt@w).clip(1e-9)[:,None]).mean(0)
            res = minimize(fun,np.full(m,1/m),jac=jac,bounds=[(0,1)]*m,
                           constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(m)},
                           method='SLSQP',options={'maxiter':200,'ftol':1e-10})
            if not res.success:
                raise RuntimeError('Simplex optimization: '+res.message)
            self.state['weights'] = normalize(res.x)
            self.params = m
        elif method.startswith('H') and method not in ('H5',):
            if method == 'H1':
                values = signed_scores(torch.tensor(p),torch.tensor(y)).numpy()
            elif method == 'H2':
                values = np.log(true.clip(1e-9))
            elif method == 'H3':
                values = -(1-true)**2
            elif method == 'H4':
                values = correct.astype(float)
            else:
                values = correct + cfg.get('rho',1.)*(correct*(~correct).mean(1,keepdims=True))
            if cfg.get('macro'):
                score = np.stack([values[y==k].mean(0) for k in np.unique(y)]).mean(0)
            else:
                score = values.mean(0)
            if method == 'H4':
                score = (score*n+1)/(n+2)
            self.state['weights'] = softmax(cfg.get('beta',1.)*score)
        elif method in ('H5','R5','R6','R7'):
            conf=p.max(-1)
            self.state['bin_x'],self.state['bin_y'] = [],[]
            for j in range(m):
                edges=np.unique(np.quantile(conf[:,j],np.linspace(0,1,6)))
                bins=np.digitize(conf[:,j],edges[1:-1])
                xx,rr=[],[]
                for k in np.unique(bins):
                    mask=bins==k
                    xx.append(float(conf[mask,j].mean()))
                    rr.append(float((correct[mask,j].sum()+1)/(mask.sum()+2)))
                self.state['bin_x'].append(np.array(xx));self.state['bin_y'].append(np.array(rr))
            self.state['switch'] = float(np.quantile(p.var(1).mean(1),.75))
        elif method in ('D5','D6','D7','D8'):
            self._fit_correctness(train,correct,seed)
        elif method == 'D9':
            from sklearn.neighbors import NearestNeighbors
            z=train.get('embedding',p.reshape(n,-1))
            self.state['knn_mean']=z.mean(0);self.state['knn_std']=np.maximum(z.std(0),1e-6)
            self.model=NearestNeighbors(n_neighbors=min(cfg.get('k',15),n)).fit((z-self.state['knn_mean'])/self.state['knn_std'])
            self.state['correct']=correct
        elif method == 'D10':
            self.state['class_reliability']=np.stack([(correct[y==k].sum(0)+1)/((y==k).sum()+2) for k in range(c)],1)
        elif method in ('R8',):
            self.state['weights']=(correct.sum(0)+1)/(n+2)
        elif method == 'S7':
            self.model=make_pipeline(StandardScaler(),LogisticRegression(C=cfg.get('C',1.),max_iter=300,random_state=seed))
            self.model.fit(p.reshape(n,-1),y)
            self.params=int(self.model[-1].coef_.size+self.model[-1].intercept_.size)
        self.train_seconds=time.perf_counter()-start
        return self

    @staticmethod
    def _static(p,y,method,cfg,epochs):
        p=torch.tensor(p,dtype=torch.float64);y=torch.tensor(y)
        a=nn.Parameter(torch.zeros((p.shape[1],p.shape[2]) if method=='S6' else (p.shape[1],),dtype=torch.float64))
        opt=torch.optim.Adam([a],lr=.05)
        errors=(p.argmax(-1)!=y[:,None]).double()
        cov=errors.T@errors/len(y)
        for _ in range(min(300,max(epochs,150))):
            opt.zero_grad();w=a.softmax(0)
            if method=='S6':
                mix=(p*w[None]).sum(1);mix=mix/mix.sum(1,keepdim=True)
                reg=(w*(w.clamp_min(1e-9).log()+np.log(len(w)))).sum(0).mean()
            else:
                mix=(p*w[None,:,None]).sum(1)
                reg=(w*(w.clamp_min(1e-9).log()+np.log(len(w)))).sum()
            losses=-mix[torch.arange(len(y)),y].clamp_min(1e-9).log()
            kind=cfg.get('loss','balanced' if method=='S3' else 'ce')
            loss=torch.stack([losses[y==k].mean() for k in y.unique()]).mean() if kind=='balanced' else losses.mean()
            if kind=='brier':loss=(mix-torch.nn.functional.one_hot(y,p.shape[-1])).square().sum(1).mean()
            loss=loss+cfg.get('kl',0.)*reg
            if method=='S5':loss=loss+cfg.get('rho',.1)*(w@cov@w)
            loss.backward();opt.step()
        return a.softmax(0).detach().numpy()

    def _fit_neural(self,train,select,seed,epochs,patience,progress_path):
        cfg=self.cfg
        if self.method=='D12':
            rng=np.random.default_rng(seed)
            d=train['embedding'].shape[-1]
            self.state['projection']=rng.normal(size=(d,min(16,d))).astype('float32')/np.sqrt(d)
        f=self.feature_data(train)
        self.state['mean']=f.mean((0,1),keepdims=True)
        self.state['std']=np.maximum(f.std((0,1),keepdims=True),1e-5)
        x=torch.tensor(self.feature_data(train));sx=torch.tensor(self.feature_data(select))
        p=torch.tensor(train['probs']);sp=torch.tensor(select['probs'])
        y=torch.tensor(train['y'],dtype=torch.long)
        capacity=cfg.get('capacity','linear' if self.method=='D1' else '64x32' if self.method=='D11' else '32x16')
        self.model=Gate(p.shape[1],x.shape[-1],capacity,joint=self.method=='D11',dropout=cfg.get('dropout',0.))
        self.params=sum(v.numel() for v in self.model.parameters())
        opt=torch.optim.AdamW(self.model.parameters(),lr=cfg.get('lr',.001),weight_decay=.001)
        best=copy.deepcopy(self.model.state_dict());best_key=(-1.,-1.,-float('inf'));stale=0;first=0
        path=Path(progress_path) if progress_path else None
        if path and path.exists():
            state=torch.load(path,map_location='cpu',weights_only=True)
            self.model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
            best=state['best'];best_key=tuple(state['best_key']);stale=state['stale'];first=state['epoch']+1
            torch.set_rng_state(state['rng'])
        for epoch in range(first,min(100,epochs)):
            self.model.train();opt.zero_grad();w=self.model(x)
            pp=p
            if self.method=='D17' or cfg.get('expert_dropout'):
                mask=(torch.rand_like(w)>.2).float();mask[:,0]=1
                w=w*mask;w=w/w.sum(1,keepdim=True)
            mix=(w[:,:,None]*pp).sum(1)
            kind=cfg.get('loss',{'D15':'focal','D16':'brier'}.get(self.method,'ce'))
            eta=cfg.get('eta',.1 if self.method=='D13' else 0.)
            tail=cfg.get('tail',.2 if self.method=='D14' else 0.)
            loss=objective(mix,w,pp,y,kind=kind,kl=cfg.get('kl',0.),eta=eta,tail=tail,gamma=cfg.get('gamma',2.))
            if not torch.isfinite(loss):raise ValueError('Nonfinite gate loss')
            loss.backward();nn.utils.clip_grad_norm_(self.model.parameters(),5.);opt.step()
            self.model.eval()
            with torch.no_grad():smix=(self.model(sx)[:,:,None]*sp).sum(1).numpy()
            met=metrics(smix,select['y']);key=(met['accuracy'],met['balanced_accuracy'],-met['nll'])
            if key>best_key:
                best_key=key;best=copy.deepcopy(self.model.state_dict());stale=0
            else:stale+=1
            if path:
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix('.tmp')
                torch.save(dict(epoch=epoch,model=self.model.state_dict(),optimizer=opt.state_dict(),best=best,
                                best_key=list(best_key),stale=stale,rng=torch.get_rng_state()),tmp)
                tmp.replace(path)
            if stale>=patience:break
        self.model.load_state_dict(best);self.model.eval()
        self.state['best_select_key']=list(best_key)

    def _fit_correctness(self,train,correct,seed):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
        f,_=features(train)
        n,m,d=f.shape
        # Expert index is an inference-time feature. OOF folds group all expert
        # rows for one sample together; no duplicate-sample calibration leakage.
        f=np.concatenate([f,np.broadcast_to(np.arange(m)[None,:,None]/m,(n,m,1))],-1)
        builders={'D5':lambda:DecisionTreeClassifier(max_depth=self.cfg.get('depth',3),min_samples_leaf=20,random_state=seed),
                  'D6':lambda:RandomForestClassifier(n_estimators=32,max_depth=self.cfg.get('depth',3),min_samples_leaf=15,n_jobs=1,random_state=seed),
                  'D7':lambda:HistGradientBoostingClassifier(max_iter=40,max_depth=self.cfg.get('depth',3),early_stopping=False,random_state=seed),
                  'D8':lambda:make_pipeline(StandardScaler(),LogisticRegression(C=self.cfg.get('C',1.),max_iter=300,random_state=seed))}
        model=builders[self.method]()
        oof=np.zeros((n,m))
        for a,b in StratifiedKFold(3,shuffle=True,random_state=seed).split(f,train['y']):
            sub=clone(model);sub.fit(f[a].reshape(-1,f.shape[-1]),correct[a].ravel())
            oof[b]=self._correct_proba(sub,f[b].reshape(-1,f.shape[-1])).reshape(len(b),m)
        self.state['correct_calibrator']=LogisticRegression(C=1.,max_iter=200).fit(oof.reshape(-1,1),correct.ravel()) if np.unique(correct).size>1 else None
        model.fit(f.reshape(-1,f.shape[-1]),correct.ravel());self.model=model

    @staticmethod
    def _correct_proba(model,x):
        proba=model.predict_proba(x)
        cls=model.classes_ if hasattr(model,'classes_') else model[-1].classes_
        return proba[:,list(cls).index(True)] if True in cls else np.zeros(len(x))

    def weights(self,data):
        data=self.calibrated(data)
        p=validate(data['probs']);n,m,c=p.shape
        if m!=self.members:raise ValueError('Ordered member count differs from fitted predictor')
        cfg,method=self.cfg,self.method
        if method in NEURAL:
            with torch.inference_mode():w=self.model(torch.tensor(self.feature_data(data))).numpy()
        elif method=='U0':w=np.full((n,m),1/m)
        elif method in ('S1','S2','S3','S4','S5','H1','H2','H3','H4','H6'):
            w=np.broadcast_to(self.state['weights'],(n,m)).copy()
        elif method in ('R1','R2','R3'):
            score=p.max(-1);k=min(cfg.get('k',1 if method=='R1' else 3),m)
            idx=np.argsort(-score,axis=1,kind='stable')[:,:k];w=np.zeros((n,m))
            values=softmax(cfg.get('beta',1.)*np.take_along_axis(score,idx,1)) if method=='R3' else np.full((n,k),1/k)
            np.put_along_axis(w,idx,values,1)
        elif method in ('H5','R5','R6','R7'):
            conf=p.max(-1)
            reliability=np.stack([np.interp(conf[:,j],self.state['bin_x'][j],self.state['bin_y'][j]) for j in range(m)],1)
            if method in ('R5','R6'):
                reliability=np.select([reliability<.5,reliability<.65,reliability<.8],[0.,.25,.5],default=1.)
            w=normalize(reliability)
            if method=='R6':
                k=min(cfg.get('k',5),m);idx=np.argsort(-reliability,axis=1,kind='stable')[:,:k]
                mask=np.zeros_like(w);np.put_along_axis(mask,idx,1.,1);w=normalize(w*mask)
            if method=='R7':w=np.where((p.var(1).mean(1)<self.state['switch'])[:,None],1/m,w)
        elif method in ('D5','D6','D7','D8'):
            f,_=features(data)
            f=np.concatenate([f,np.broadcast_to(np.arange(m)[None,:,None]/m,(n,m,1))],-1)
            rel=self._correct_proba(self.model,f.reshape(-1,f.shape[-1]))
            calibrator=self.state['correct_calibrator']
            if calibrator is not None:rel=calibrator.predict_proba(rel[:,None])[:,1]
            w=normalize(rel.reshape(n,m))
        elif method=='D9':
            z=data.get('embedding',p.reshape(n,-1));z=(z-self.state['knn_mean'])/self.state['knn_std']
            idx=self.model.kneighbors(z,return_distance=False)
            w=normalize((self.state['correct'][idx].sum(1)+1)/(idx.shape[1]+2))
        elif method=='D10':w=normalize(p.mean(1)@self.state['class_reliability'].T)
        else:return None
        if cfg.get('static_average') and 'static_average' in self.state:
            w=np.broadcast_to(self.state['static_average'],(n,m)).copy()
        k=cfg.get('top_k',5 if method=='D18' else None)
        if k is not None and k<m:
            idx=np.argsort(-w,axis=1,kind='stable')[:,:k];mask=np.zeros_like(w)
            np.put_along_axis(mask,idx,1.,1);w=normalize(w*mask)
        if cfg.get('random_weights'):
            # Hash-like deterministic sample-local input independent of labels.
            rng=np.random.default_rng(193);w=np.broadcast_to(normalize(rng.random(m)),(n,m)).copy()
        if cfg.get('weight_binning'):
            w=np.round(w*m)/m
        return normalize(w)

    def predict(self,data):
        """No labels or correctness are accepted as arguments or gate inputs."""
        p=self.calibrated(data)['probs'];n,m,c=p.shape
        if self.method=='S7':return self.model.predict_proba(p.reshape(n,-1))
        if self.method=='S6':
            mix=(p*self.state['weights'][None]).sum(1)
            return mix/mix.sum(1,keepdims=True)
        if self.method in ('R4','R8'):
            if self.method=='R4' and self.cfg.get('variant')=='median':
                return normalize(np.median(p,axis=1))
            votes=np.eye(c)[p.argmax(-1)]
            if self.method=='R8':votes=votes*self.state['weights'][None,:,None]
            # Small mean-probability contribution breaks ties deterministically.
            return normalize(votes.sum(1)+1e-8*p.mean(1))
        w=self.weights(data)
        return (w[:,:,None]*p).sum(1)
