import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def clusters(embeddings):
    pca=PCA(24,svd_solver='randomized',random_state=0)
    z=pca.fit_transform(embeddings)
    km=KMeans(n_clusters=2,n_init=6,random_state=1).fit(z)
    return pca,km,km.labels_


def solve(p,y,cluster,epigraph=True):
    a=p[np.arange(len(y))[:,None],np.arange(15),y[:,None]].astype(float)
    u=np.full(15,1/15)
    pools=[a[cluster==c] for c in (0,1)]
    if any(len(v)==0 for v in pools):raise ValueError('Empty latent loss cluster')
    def losses(w):return np.array([-np.log((v@w).clip(1e-8,1)).mean() for v in pools])
    def gradients(w):
        return np.array([-(v/(v@w).clip(1e-8,1)[:,None]).mean(0) for v in pools])
    if epigraph:
        def fun(x):return x[-1]+.02*((x[:15]-u)**2).sum()
        def jac(x):return np.r_[.04*(x[:15]-u),1.]
        constraints=[dict(type='eq',fun=lambda x:x[:15].sum()-1,jac=lambda x:np.r_[np.ones(15),0.]),
                     dict(type='ineq',fun=lambda x:x[-1]-losses(x[:15]),jac=lambda x:np.column_stack([-gradients(x[:15]),np.ones(2)]))]
        result=minimize(fun,np.r_[u,max(losses(u))],jac=jac,bounds=[(0,.25)]*15+[(None,None)],constraints=constraints,method='SLSQP',options=dict(maxiter=1000,ftol=1e-12))
        w=result.x[:15];slack=result.x[-1]-losses(w)
    else:
        result=minimize(lambda w:max(losses(w))+.02*((w-u)**2).sum(),u,
                        bounds=[(0,.25)]*15,constraints=[dict(type='eq',fun=lambda w:w.sum()-1)],method='SLSQP',options=dict(maxiter=1000,ftol=1e-12))
        w=result.x;slack=np.zeros(2)
    audit=dict(success=bool(result.success),message=str(result.message),objective=float(max(losses(w))+.02*((w-u)**2).sum()),
               cluster_losses=losses(w).tolist(),simplex_residual=float(abs(w.sum()-1)),min_weight=float(w.min()),max_weight=float(w.max()),
               min_epigraph_slack=float(slack.min()),iterations=int(result.nit))
    if not result.success or abs(w.sum()-1)>1e-7 or w.min() < -1e-8 or w.max()>.25+1e-8 or slack.min() < -1e-7:raise RuntimeError(str(audit))
    return w,audit
