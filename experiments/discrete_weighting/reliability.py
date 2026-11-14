import numpy as np


def softmax(z):
    e=np.exp(z-z.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)


def calibrate(p,t):return softmax(np.log(p.clip(1e-12))/np.asarray(t)[None,:,None])


def fit_temperature(p,y,shared=False):
    grid=np.geomspace(.03,8,41)
    losses=np.array([-np.log(calibrate(p,np.full(p.shape[1],t))[np.arange(len(y))[:,None],np.arange(p.shape[1]),y[:,None]].clip(1e-12)).mean(0) for t in grid])
    return np.full(p.shape[1],grid[losses.mean(1).argmin()]) if shared else grid[losses.argmin(0)]


class Rule:
    def __init__(self,cfg):self.cfg=dict(cfg)

    def feature(self,p):
        n,m,_=p.shape;pred=p.argmax(-1);conf=p.max(-1)
        agreement=((pred[:,:,None]==pred[:,None,:]).sum(-1)-1)/max(1,m-1)
        if self.cfg.get('rank'):conf=np.argsort(np.argsort(conf,axis=1,kind='stable'),axis=1,kind='stable')/max(1,m-1)
        return pred,conf,agreement

    def fit(self,p,y,temperature=None):
        c=self.cfg;m=p.shape[1];self.temperature=np.ones(m) if temperature is None else np.asarray(temperature)
        pp=calibrate(p,self.temperature);pred,conf,agree=self.feature(pp)
        self.edges=np.zeros((2,2))
        for k in (0,1):
            v=conf[pred==k] if c.get('condition',True) else conf.ravel();self.edges[k]=np.quantile(v,[1/3,2/3]) if len(v) else [.5,.5]
        pc,cb,ab=self.bins(pp);correct=pred==y[:,None]
        sample_w=np.ones(len(y))
        if c.get('balance')=='true':sample_w=np.array([len(y)/(2*max(1,(y==v).sum())) for v in y])
        observation_w=np.broadcast_to(sample_w[:,None],pred.shape).copy()
        if c.get('balance')=='predicted':
            for k in (0,1):observation_w[pred==k]=pred.size/(2*max(1,(pred==k).sum()))
        global_prior=float((correct*observation_w).sum()/observation_w.sum())
        self.pool=np.zeros((2,3,3));self.table=np.zeros((m,2,3,3));self.counts=np.zeros_like(self.table)
        for k in range(2):
            coarse=pc==k
            prior=((correct[coarse]*observation_w[coarse]).sum()+2*global_prior)/(observation_w[coarse].sum()+2)
            for b in range(3):
                mask=coarse&(cb==b);count=observation_w[mask].sum()
                confidence_prior=((correct[mask]*observation_w[mask]).sum()+2*prior)/(count+2)
                for a in range(3):
                    cell=mask&(ab==a);total=observation_w[cell].sum()
                    pooled=((correct[cell]*observation_w[cell]).sum()+2*confidence_prior)/(total+2) if total>=c.get('min_count',10) else confidence_prior
                    self.pool[k,b,a]=pooled
                    for j in range(m):
                        q=cell[:,j];num=observation_w[q,j].sum();self.counts[j,k,b,a]=num
                        kap=c.get('kappa',50)
                        self.table[j,k,b,a]=((correct[q,j]*observation_w[q,j]).sum()+kap*pooled)/(num+kap) if num+kap else pooled
        self.guard_margin=float(np.quantile(np.abs(pp.mean(1)[:,1]-.5)*2,c.get('guard_quantile',.5)))
        self.static=(correct.sum(0)+1)/(len(y)+2)
        self.threshold=.5
        if c.get('family')=='D1':
            candidates=np.unique(np.r_[.5,np.quantile(pp.mean(1)[:,1],np.linspace(.05,.95,31))])
            self.threshold=float(max(candidates,key=lambda t:(((pp.mean(1)[:,1]>t)==y).mean(),-abs(t-.5))))
        self.train_average=None
        if c.get('static_average'):self.train_average=self.weights(p).mean(0)
        return self

    def bins(self,p):
        pred,conf,agree=self.feature(p);pc=pred if self.cfg.get('condition',True) else np.zeros_like(pred)
        cb=np.zeros_like(pred) if self.cfg.get('confidence',True) is False else np.stack([(conf[:,j]>self.edges[pc[:,j],0]).astype(int)+(conf[:,j]>self.edges[pc[:,j],1]).astype(int) for j in range(p.shape[1])],1)
        ab=np.digitize(agree,[1/3,2/3],right=True) if self.cfg.get('agreement') else np.zeros_like(pred)
        return pc,cb,ab

    def weights(self,p):
        pp=calibrate(p,self.temperature);n,m,_=p.shape;c=self.cfg
        if c.get('family') in ('D0a','D0b','D1'):return np.full((n,m),1/m)
        pc,cb,ab=self.bins(pp)
        r=self.table[np.arange(m)[None,:],pc,cb,ab] if c.get('head') else self.pool[pc,cb,ab]
        if c.get('family')=='C0':r=np.broadcast_to(self.static,(n,m))
        if c.get('random'):r=np.random.default_rng(271).random((n,m))
        if c.get('shuffle'):r=np.roll(r,1,axis=1)
        score=2*r-1
        if c.get('ordinal'):score=np.select([r<.5,r<.65,r<.8,r<.95],[-1.,-.5,0.,.5],default=1.)
        v=softmax(c.get('beta',3)*score)
        k=min(c.get('top_k',m),m)
        if k<m:
            mask=np.zeros_like(v);np.put_along_axis(mask,np.argsort(-r,axis=1,kind='stable')[:,:k],1,axis=1)
            v=v*mask;v=v/v.sum(1,keepdims=True)
        alpha=c.get('alpha',1.);w=(1-alpha)/m+alpha*v
        bad=(~np.isfinite(r).all(1))|(r.max(1)<=.5);w[bad]=1/m
        if c.get('guard'):
            base=pp.mean(1);pred=base.argmax(1);votes=(pp.argmax(-1)==pred[:,None]).sum(1)
            strong=(votes>=c.get('guard_votes',12)*m/15)&(np.abs(base[:,1]-.5)*2>=self.guard_margin)
            w[strong]=1/m
        if getattr(self,'train_average',None) is not None:w=np.broadcast_to(self.train_average,(n,m)).copy()
        return w

    def predict(self,p):
        w=self.weights(p);return (calibrate(p,self.temperature)*w[:,:,None]).sum(1),w


def recipes():
    configs=[]
    def add(f,**kw):configs.append(dict(family=f,**kw))
    for f in ('D0a','D0b','D1'):add(f,calibration='head' if f=='D0b' else 'raw')
    add('D2');
    for k in (10,50,200):add('D3',head=True,kappa=k)
    for alpha,beta in ((.1,1),(.25,3),(.5,5),(.75,3)):
        add('D4',head=True,alpha=alpha,beta=beta)
        add('D5',head=True,agreement=True,alpha=alpha,beta=beta)
    add('D6',head=True,agreement=True,guard=True,alpha=.5)
    add('D7',head=True,ordinal=True,alpha=.5)
    add('D8',head=True,ordinal=False,alpha=.5)
    add('D9',head=True,balance='predicted',alpha=.5)
    add('D9',head=True,balance='true',alpha=.5)
    for k in (3,5,10,15):add('D10',head=True,top_k=k,alpha=.5,guard=True)
    add('D11',head=True,rank=True,alpha=.5)
    for cal in ('raw','head','shared'):add('D12',head=True,alpha=.5,calibration=cal)
    add('C0',alpha=.5)
    return configs


def ablations(winner):
    specs=[('A0','full',{})]
    for cal in ('raw','head'):specs.append(('A1',cal,dict(family='D0a' if cal=='raw' else 'D0b',calibration=cal)))
    for a,label,change in [('A2','no_class',dict(condition=False)),('A3','no_confidence',dict(confidence=False)),('A4','no_agreement',dict(agreement=False)),('A5','no_guard',dict(guard=False)),('A6','pooled',dict(head=False)),('A6','head',dict(head=True)),('A7','unsmoothed',dict(kappa=0)),('A9','ordinal',dict(ordinal=True)),('A9','continuous',dict(ordinal=False)),('A13','balanced',dict(balance='true')),('A13','empirical',dict(balance=None)),('A15','static',dict(static_average=True)),('A20','random',dict(random=True)),('A20','shuffled_bins',dict(shuffle=True))]:specs.append((a,label,change))
    for a,field,values in [('A8','kappa',[10,50,200]),('A10','alpha',[.1,.25,.5,1.]),('A11','beta',[.5,1,2,5,10]),('A12','calibration',['raw','head','shared']),('A14','top_k',[3,5,10,15]),('A16','members',[3,5,10,15]),('A17','guard_votes',[10,12,14]),('A18','folds',[3,5]),('A19','split_seed',[200,201])]:
        for value in values:specs.append((a,str(value),{field:value}))
    return [dict(ablation=a,variant=l,config=winner|v) for a,l,v in specs]
