"""B mechanism audit and post-inference C evaluation; never select using C."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from experiments.geometry_aggregation_v5.io import load,save_npz
from experiments.geometry_aggregation_v5.persistence import atomic
from experiments.geometry_aggregation_v5.journal import csvwrite
from experiments.discrete_weighting.store import sha
from .store import ROOT,ROLE,COMMAND,digest,jsonwrite
from .model import METHODS,target,center
from .run import B,C,outer_splits,folder,pred_path,PRODUCTION,V6


def metric(p,y,group=None):
    y=np.asarray(y,int);correct=p.argmax(-1)==y;loss=-np.log(p[np.arange(len(y)),y].clip(1e-15))
    ca=[float(correct[y==k].mean()) for k in range(2)];cl=[float(loss[y==k].mean()) for k in range(2)]
    out=dict(n=len(y),AA=float(correct.mean()),NLL=float(loss.mean()),worst_class_NLL=max(cl),class0_NLL=cl[0],class1_NLL=cl[1],
        WCA=min(ca),BA=float(np.mean(ca)),class0_accuracy=ca[0],class1_accuracy=ca[1])
    if group is not None:
        assert set(np.unique(group))=={0,1,2,3};ga=[float(correct[group==g].mean()) for g in range(4)]
        out.update(WGA=min(ga),group_accuracy=ga,group_count=[int((group==g).sum()) for g in range(4)],worst_groups=[g for g in range(4) if ga[g]==min(ga)])
    return out


def ranks(a,t):
    ar=rankdata(a,axis=1);tr=rankdata(t,axis=1);ar=center(ar);tr=center(tr);den=np.sqrt((ar*ar).sum(1)*(tr*tr).sum(1))
    rho=np.divide((ar*tr).sum(1),den,out=np.full(len(a),np.nan),where=den>0)
    mask=(t[:,:,None]-t[:,None,:])>1e-15;count=mask.sum((1,2));diff=a[:,:,None]-a[:,None,:]
    wins=((diff>0)+.5*(diff==0))*mask
    pair=np.divide(wins.sum((1,2)),count,out=np.full(len(a),np.nan),where=count>0)
    return rho,pair


def finite(x):return float(x) if np.isfinite(x) else None


def diagnostics(s,f,part,method,out,member,y,ids,model,g1):
    n=len(y);t=target(member,y);gain=(out['r']*t).sum(1);gg=(g1['r']*t).sum(1)
    pred=out['probs'].argmax(1);up=out['uniform'].argmax(1);gp=g1['probs'].argmax(1);margin=abs(out['uniform'][:,1]-.5)
    w=out['weights'];geom=center(out['geom_scores']);res=out['residual_scores'];tv=.5*abs(w-g1['weights']).sum(1)
    estimate=out['geom_scores'] if method=='G1_frozen' else out['Ahat'];rho,pair=ranks(estimate,t);grho,gpair=ranks(out['geom_scores'],t);prho,ppair=ranks(out['pooled_Ahat'],t)
    scales=model['scales']['pooled' if method=='G1_R' else 'head'];sR=scales['s_R'];clipfrac=(abs(center(out['Ahat']))>2*sR).mean(1) if sR and scales['enabled'] else np.zeros(n)
    rows=[]
    for i in range(n):
        rows.append(dict(seed=s,outer_fold=f,inner_fold=part,method=method,sample_id=str(ids[i]),y=int(y[i]),
            p0=float(out['probs'][i,0]),p1=float(out['probs'][i,1]),uniform_p0=float(out['uniform'][i,0]),uniform_p1=float(out['uniform'][i,1]),
            G1_p0=float(g1['probs'][i,0]),G1_p1=float(g1['probs'][i,1]),prediction=int(pred[i]),uniform_wrong=bool(up[i]!=y[i]),
            gain=float(gain[i]),gain_G1=float(gg[i]),gain_difference_G1=float(gain[i]-gg[i]),gain_positive=bool(gain[i]>0),
            G1_negative_to_positive=bool(gg[i]<0 and gain[i]>0),true_probability_shift=float(out['probs'][i,y[i]]-out['uniform'][i,y[i]]),
            corrected_U0=bool(pred[i]==y[i] and up[i]!=y[i]),broken_U0=bool(pred[i]!=y[i] and up[i]==y[i]),
            corrected_G1=bool(pred[i]==y[i] and gp[i]!=y[i]),broken_G1=bool(pred[i]!=y[i] and gp[i]==y[i]),
            rank_spearman=finite(rho[i]),pair_ranking=finite(pair[i]),geometry_rank_spearman=finite(grho[i]),geometry_pair_ranking=finite(gpair[i]),
            pooled_rank_spearman=finite(prho[i]),pooled_pair_ranking=finite(ppair[i]),weight_TV=float(tv[i]),
            max_weight_change=float(abs(w[i]-g1['weights'][i]).max()),top_head=int(w[i].argmax()),top_changed=bool(w[i].argmax()!=g1['weights'][i].argmax()),
            rmax=float(out['r'][i].max()),entropy=float(-(w[i]*np.log(w[i].clip(1e-15))).sum()),effective_experts=float(1/(w[i]**2).sum()),
            residual_range=float(np.ptp(res[i])),residual_RMS=float(np.sqrt(np.mean(res[i]**2))),residual_SD=float(np.std(res[i])),
            geometry_range=float(np.ptp(geom[i])),geometry_RMS=float(np.sqrt(np.mean(geom[i]**2))),
            score_RMS_ratio=float(np.sqrt(np.mean(center(res[i:i+1])**2))/max(np.sqrt(np.mean(geom[i]**2)),1e-15)),
            clip_fraction=float(clipfrac[i]),gate=float(out['gate'][i]),uniform_margin=float(margin[i]),
            margin_stratum='low' if margin[i]<=model['margin_quantiles'][0] else 'high' if margin[i]>model['margin_quantiles'][2] else 'mid',
            s_R=sR,s_G=scales['s_G'],scale_enabled=bool(scales['enabled']),fallback_reason=str(out['fallback_reason']),
            prediction_cache=pred_path(s,f,part,method).as_posix(),role=ROLE))
    return rows


def B_evaluate(s,f,part):
    b=B(s,f);v=outer_splits(b,s,f)[part][1];model=json.loads((folder(s,f,part)/'model.json').read_text());rows=[];stats=[]
    g1=load(pred_path(s,f,part,'G1_frozen'))
    for m in METHODS:
        out=load(pred_path(s,f,part,m));np.testing.assert_array_equal(out['sample_id'],b['sample_id'][v])
        assert not set(model['fit_ids'])&set(out['sample_id'])
        rows+=diagnostics(s,f,part,m,out,b['probs'][v],b['y'][v],out['sample_id'],model,g1)
        stats.append(dict(seed=s,outer_fold=f,inner_fold=part,method=m,**metric(out['probs'],b['y'][v]),role=ROLE))
    p=folder(s,f,part)/'B_validation_diagnostics.csv';csvwrite(p,rows);q=p.with_name('B_validation_metrics.json');jsonwrite(q,stats)
    return dict(seed=s,fold=f,inner=part,images=len(v),methods=5),[p,q]


def B_tables():
    df=pd.concat([pd.read_csv(folder(s,f,part)/'B_validation_diagnostics.csv') for s in range(3) for f in range(3) for part in range(4)],ignore_index=True)
    assert 'group' not in df;paths=[]
    p=ROOT/'metrics/B_OOF_MECHANISM.csv';csvwrite(p,df.replace({np.nan:None}).to_dict('records'));paths.append(p)
    cols=['seed','outer_fold','inner_fold','method','sample_id','y','uniform_wrong','gain','gain_G1','gain_difference_G1','gain_positive','G1_negative_to_positive',
        'corrected_U0','broken_U0','corrected_G1','broken_G1','true_probability_shift','rank_spearman','pair_ranking','weight_TV','role']
    p=ROOT/'metrics/B_OOF_SIGNED_GAIN.csv';csvwrite(p,df[cols].replace({np.nan:None}).to_dict('records'));paths.append(p)
    ensemble=[]
    for s in range(3):
        for f in range(3):
            for m in METHODS:
                z=df[(df.seed==s)&(df.outer_fold==f)&(df.method==m)];assert len(z)==len(set(z.sample_id))==240
                ensemble.append(dict(seed=s,outer_fold=f,method=m,**metric(z[['p0','p1']].to_numpy(),z.y.to_numpy()),role=ROLE))
    p=ROOT/'metrics/B_ENSEMBLE_METRICS.csv';csvwrite(p,ensemble);paths.append(p)
    # Scalar mechanism statistics are averaged within image; expert slots are
    # never averaged/transferred between different actual ensembles.
    keys=['seed','method','sample_id'];numeric=['y','p0','p1','uniform_p0','uniform_p1','G1_p0','G1_p1','gain','gain_G1','gain_difference_G1',
        'rank_spearman','pair_ranking','geometry_rank_spearman','geometry_pair_ranking','weight_TV','score_RMS_ratio','top_changed','residual_RMS','geometry_RMS','gate']
    pooled=df.groupby(keys,sort=True)[numeric].mean().reset_index();pooled['multiplicity']=df.groupby(keys).size().to_numpy()
    assert all(df.groupby(keys).y.nunique()==1)
    pooled['y']=pooled.y.astype(int);pooled['uniform_wrong']=(pooled.uniform_p1>pooled.uniform_p0).astype(int)!=pooled.y
    pooled['prediction']=(pooled.p1>pooled.p0).astype(int);pooled['correct']=pooled.prediction==pooled.y
    pooled['uniform_correct']=~pooled.uniform_wrong;pooled['G1_correct']=((pooled.G1_p1>pooled.G1_p0).astype(int)==pooled.y)
    p=ROOT/'metrics/B_SEED_UNIQUE_IMAGE_PREDICTIONS.csv';csvwrite(p,pooled.replace({np.nan:None}).to_dict('records'));paths.append(p)
    seeds=[]
    for s in range(3):
        for m in METHODS:
            z=pooled[(pooled.seed==s)&(pooled.method==m)];u=pooled[(pooled.seed==s)&(pooled.method=='U0')];g=pooled[(pooled.seed==s)&(pooled.method=='G1_frozen')]
            r=metric(z[['p0','p1']].to_numpy(),z.y.to_numpy());wrong=z[z.uniform_wrong]
            r.update(seed=s,method=m,unique_images=len(z),raw_rows=int(z.multiplicity.sum()),uniform_wrong_images=len(wrong),
                gain_difference_Uwrong=finite(wrong.gain_difference_G1.mean()),mean_gain_Uwrong=finite(wrong.gain.mean()),
                gain_positive_fraction_Uwrong=finite((wrong.gain>0).mean()),G1_negative_to_positive_Uwrong=int(((wrong.gain_G1<0)&(wrong.gain>0)).sum()),
                delta_AA_U0_pp=100*(r['AA']-u.correct.mean()),delta_AA_G1_pp=100*(r['AA']-g.correct.mean()),
                corrected_U0=int((z.correct&~z.uniform_correct).sum()),broken_U0=int((~z.correct&z.uniform_correct).sum()),
                corrected_G1=int((z.correct&~z.G1_correct).sum()),broken_G1=int((~z.correct&z.G1_correct).sum()),
                mean_weight_TV=float(z.weight_TV.mean()),rank_spearman_Uwrong=finite(wrong.rank_spearman.mean()),
                geometry_rank_Uwrong=finite(wrong.geometry_rank_spearman.mean()),pair_ranking_Uwrong=finite(wrong.pair_ranking.mean()),
                geometry_pair_ranking_Uwrong=finite(wrong.geometry_pair_ranking.mean()),role=ROLE)
            seeds.append(r)
    p=ROOT/'metrics/B_SEED_METRICS.csv';csvwrite(p,seeds);paths.append(p)
    margin=[]
    for (m,stratum),z in df.groupby(['method','margin_stratum']):
        margin.append(dict(method=m,margin_stratum=stratum,n=len(z),unique_images=z.sample_id.nunique(),AA=float((z.prediction==z.y).mean()),
            mean_gate=float(z.gate.mean()),corrected_U0=int(z.corrected_U0.sum()),broken_U0=int(z.broken_U0.sum()),
            corrected_G1=int(z.corrected_G1.sum()),broken_G1=int(z.broken_G1.sum()),weight_TV=float(z.weight_TV.mean()),role=ROLE))
    p=ROOT/'metrics/B_MARGIN_DIAGNOSTICS.csv';csvwrite(p,margin);paths.append(p)
    cells=[];stability=[]
    for s in range(3):
        for f in range(3):
            models=[json.loads((folder(s,f,part)/'model.json').read_text()) for part in range(4)]
            for part,model in enumerate(models):cells.extend(dict(seed=s,outer_fold=f,inner_fold=part,**c,role=ROLE) for c in model['cells'])
            values=np.array([m['head_effects'] for m in models])
            for j in range(15):
                for k in range(2):
                    v=values[:,j,k];stability.append(dict(seed=s,outer_fold=f,head=j,head_sha256=models[0]['head_sha256'][j],predicted_class=k,
                        mean_b=float(v.mean()),std_b=float(v.std()),positive_fits=int((v>0).sum()),negative_fits=int((v<0).sum()),
                        sign_agreement=float(max((v>0).sum(),(v<0).sum())/4),role=ROLE))
    p=ROOT/'metrics/B_HEAD_EFFECTS.csv';csvwrite(p,cells);paths.append(p);p=ROOT/'metrics/B_HEAD_EFFECT_STABILITY.csv';csvwrite(p,stability);paths.append(p)
    return dict(B_rows=len(df),seed_unique_images=[int(pooled[(pooled.seed==s)&(pooled.method=='U0')].shape[0]) for s in range(3)],methods=5),paths


def rng_for(kind,batch,seed):
    rng=np.random.default_rng(seed)
    if batch:rng.bit_generator.state=json.loads((ROOT/f'metrics/{kind}/batch{batch-1:03d}.json').read_text())['rng_after']
    return rng


def shared_draws(strata,rng,repeats):
    """One ID draw per replicate shared by every seed and every method."""
    return np.concatenate([rng.choice(np.flatnonzero(strata==k),(repeats,int((strata==k).sum())),replace=True) for k in np.unique(strata)],axis=1)


def draw_receipt(kind,batch,before,rng,hashes):
    p=ROOT/f'metrics/{kind}/batch{batch:03d}.json';jsonwrite(p,dict(role=ROLE,batch=batch,rng_before=before,rng_after=rng.bit_generator.state,
        shared_ID_draw_hashes=hashes,unit='unique image ID; same draws/permutations reproducible across resume',repeats=len(hashes)))
    return p


def B_bootstrap(batch):
    df=pd.read_csv(ROOT/'metrics/B_SEED_UNIQUE_IMAGE_PREDICTIONS.csv');ids=np.array(sorted(df.sample_id.unique()));idx={v:i for i,v in enumerate(ids)};N=len(ids)
    labels=np.zeros(N,int);gain=np.zeros((5,3,N));aa=np.zeros_like(gain);present=np.zeros((3,N));wrong=np.zeros_like(present)
    for s in range(3):
        for mi,m in enumerate(METHODS):
            z=df[(df.seed==s)&(df.method==m)];take=np.array([idx[v] for v in z.sample_id]);labels[take]=z.y
            gain[mi,s,take]=z.gain_difference_G1;aa[mi,s,take]=z.correct.astype(float)-z.G1_correct.astype(float)
            present[s,take]=1;wrong[s,take]=z.uniform_wrong.astype(float)
    rng=rng_for('B_bootstrap',batch,735001);before=rng.bit_generator.state
    draws=shared_draws(labels,rng,100)
    counts=np.zeros((100,N),int)
    for i,row in enumerate(draws):counts[i]=np.bincount(row,minlength=N)
    outgain=np.zeros((5,3,100));outaa=np.zeros_like(outgain)
    for mi in range(5):
        for s in range(3):
            den=counts@wrong[s];outgain[mi,s]=np.divide(counts@(gain[mi,s]*wrong[s]),den,out=np.full(100,np.nan),where=den>0)
            outaa[mi,s]=100*(counts@aa[mi,s])/(counts@present[s])
    p=ROOT/f'metrics/B_bootstrap/batch{batch:03d}.npz';save_npz(p,gain_difference_G1_Uwrong=outgain,delta_AA_G1_pp=outaa,role=np.array(ROLE))
    hashes=[digest(sorted(ids[row].tolist())) for row in draws];q=draw_receipt('B_bootstrap',batch,before,rng,hashes)
    return dict(batch=batch,repeats=100,unique_images=N),[p,q]


def B_gate():
    seeds=pd.read_csv(ROOT/'metrics/B_SEED_METRICS.csv');ensemble=pd.read_csv(ROOT/'metrics/B_ENSEMBLE_METRICS.csv');primary=seeds[seeds.method=='G1_RH']
    deltas=primary.gain_difference_Uwrong.to_numpy();mean=float(deltas.mean());positive=int((deltas>0).sum());floor=bool((primary.delta_AA_U0_pp> -1).all())
    proxy=ensemble.groupby('method').worst_class_NLL.mean();proxy_delta=float(proxy['G1_RH']-proxy['G1_frozen']);direction=bool(mean>0 and positive>=2)
    boot=np.concatenate([load(ROOT/f'metrics/B_bootstrap/batch{i:03d}.npz')['gain_difference_G1_Uwrong'] for i in range(20)],axis=-1)
    ci=np.nanquantile(np.nanmean(boot[3],axis=0),[.025,.975]).tolist();rows=[]
    for mi,m in enumerate(METHODS):
        rows.append(dict(method=m,gain_difference_Uwrong_mean=float(seeds[seeds.method==m].gain_difference_Uwrong.mean()),
            CI95=np.nanquantile(np.nanmean(boot[mi],axis=0),[.025,.975]).tolist(),role=ROLE))
    result=dict(status='B_DIRECTION_GO' if direction and floor and proxy_delta<=.003 else 'B_DIRECTION_NO_GO',
        primary='G1_RH',mean_gain_difference_Uwrong=mean,per_seed_gain_difference=deltas.tolist(),positive_seeds=positive,direction_pass=direction,
        AA_floor_every_seed=floor,per_seed_delta_AA_U0_pp=primary.delta_AA_U0_pp.tolist(),per_seed_delta_AA_G1_pp=primary.delta_AA_G1_pp.tolist(),
        worst_class_NLL_delta=proxy_delta,proxy_tolerance=.003,proxy_pass=proxy_delta<=.003,mean_worst_class_NLL=proxy.to_dict(),
        gain_CI95=ci,uncertainty='INCONCLUSIVE' if ci[0]<=0<=ci[1] else 'DIRECTIONAL_IMAGE_BOOTSTRAP_INTERVAL',
        C_permitted_regardless=True,promotion=False,role=ROLE)
    p=ROOT/'metrics/B_DIRECTION_DECISION.json';jsonwrite(p,result);q=ROOT/'metrics/B_BOOTSTRAP_CI.json';jsonwrite(q,rows)
    return result,[p,q]


def C_pool(s,method):
    parts=[load(pred_path(s,f,4,method)) for f in range(3)];ids=np.concatenate([v['sample_id'] for v in parts]);assert len(ids)==len(set(ids))==1199
    order=np.argsort(ids);keys=['probs','uniform','r','weights','Ahat','pooled_Ahat','geom_scores','residual_scores','scores','delta','z','gate']
    out={k:np.concatenate([v[k] for v in parts])[order] for k in keys};out['sample_id']=ids[order]
    out['fold']=np.concatenate([np.full(len(p['sample_id']),f) for f,p in enumerate(parts)])[order]
    return out


def C_labels(s):
    parts=[C(s,f,True) for f in range(3)];ids=np.concatenate([v['sample_id'] for v in parts]);order=np.argsort(ids)
    return {k:np.concatenate([v[k] for v in parts])[order] for k in ['sample_id','y','group']}


def C_evaluate(s,method):
    z=C_pool(s,method);labels=C_labels(s);np.testing.assert_array_equal(z['sample_id'],labels['sample_id'])
    r=metric(z['probs'],labels['y'],labels['group']);assert r['group_count']==[467,466,133,133]
    r.update(seed=s,method=method,role=ROLE);p=ROOT/f'metrics/C_seed/seed{s}/{method}.json';jsonwrite(p,r)
    return dict(seed=s,method=method,AA=r['AA'],WGA=r['WGA']),[p]


def C_export():
    pools=[[C_pool(s,m) for s in range(3)] for m in METHODS];reference=pools[0][0];labels=C_labels(0)
    for s in range(3):
        lab=C_labels(s)
        for k in ['sample_id','y','group']:np.testing.assert_array_equal(labels[k],lab[k])
        for mi in range(5):np.testing.assert_array_equal(pools[mi][s]['sample_id'],reference['sample_id'])
    keys=['probs','uniform','r','weights','Ahat','pooled_Ahat','geom_scores','residual_scores','scores','delta','z','gate']
    arrays={k:np.stack([[p[k] for p in row] for row in pools]) for k in keys}
    arrays.update(sample_id=reference['sample_id'],fold=np.stack([p['fold'] for p in pools[0]]),seeds=np.arange(3),methods=np.array(METHODS),classes=np.arange(2),split=np.array('C'),role=np.array(ROLE))
    members={k:[] for k in ['probs','distances']}
    for s in range(3):
        parts=[C(s,f) for f in range(3)];ids=np.concatenate([v['sample_id'] for v in parts]);order=np.argsort(ids)
        for k in members:members[k].append(np.concatenate([v[k] for v in parts])[order])
    arrays.update(member_probs=np.stack(members['probs']),distances=np.stack(members['distances']),
        head_sha256=np.array([[C(s,f)['head_sha256'] for f in range(3)] for s in range(3)]),
        B_model_sha256=np.array([[sha(folder(s,f,4)/'model.json') for f in range(3)] for s in range(3)]),
        probability_shift_G1=arrays['probs']-arrays['probs'][1:2],probability_shift_U0=arrays['probs']-arrays['probs'][0:1])
    p=ROOT/'metrics/WEIGHT_AND_SCORE_DECOMPOSITION.npz';save_npz(p,**arrays)
    q=ROOT/'metrics/C_EVALUATION_LABELS.npz';save_npz(q,**labels,role=np.array(ROLE))
    schema=ROOT/'metrics/CACHE_SCHEMA.json';jsonwrite(schema,dict(role=ROLE,shapes={k:list(v.shape) for k,v in arrays.items()},
        axes='method, seed, canonical sorted image ID, expert/class',head_order='head_sha256[seed, fold[seed,image], expert]',
        evaluation_labels='C_EVALUATION_LABELS.npz only; not supplied to any model/inference API'))
    return dict(samples=1199,methods=5,seeds=3,label_free_decomposition=True),[p,q,schema]


def C_bootstrap(batch):
    z=load(ROOT/'metrics/WEIGHT_AND_SCORE_DECOMPOSITION.npz');labels=load(ROOT/'metrics/C_EVALUATION_LABELS.npz');correct=z['probs'].argmax(-1)==labels['y']
    rng=rng_for('C_bootstrap',batch,730001);before=rng.bit_generator.state;ga=[];draws=shared_draws(labels['group'],rng,100);counts=[];start=0
    for g in range(4):
        idx=np.flatnonzero(labels['group']==g);draw=draws[:,start:start+len(idx)];start+=len(idx);counts.append(len(idx));ga.append(correct[:,:,draw].mean(-1))
    group_accuracy=np.stack(ga,-1);AA=(group_accuracy*np.array(counts)).sum(-1)/sum(counts);WGA=group_accuracy.min(-1)
    p=ROOT/f'metrics/C_bootstrap/batch{batch:03d}.npz';save_npz(p,AA=AA,WGA=WGA,group_accuracy=group_accuracy,role=np.array(ROLE))
    hashes=[digest(sorted(z['sample_id'][v].tolist())) for v in draws];q=draw_receipt('C_bootstrap',batch,before,rng,hashes)
    return dict(batch=batch,repeats=100,shared_unique_IDs=True),[p,q]


def C_permutation(batch):
    z=load(ROOT/'metrics/WEIGHT_AND_SCORE_DECOMPOSITION.npz');lab=load(ROOT/'metrics/C_EVALUATION_LABELS.npz');rng=rng_for('C_permutation',batch,740001);before=rng.bit_generator.state
    AA=np.zeros((2,3,50));WGA=np.zeros_like(AA);concentration=np.zeros_like(AA);gain=np.zeros_like(AA);ga=np.zeros((2,3,50,4));hashes=[]
    geom=z['geom_scores'][3];res=z['residual_scores'][3];member=z['member_probs'];delta=z['delta'][3];t=(2*lab['y']-1)[None,:,None]*delta
    wrong=z['probs'][0].argmax(-1)!=lab['y'];base_w=z['weights'][1]
    for repeat in range(50):
        perm=rng.random(res.shape).argsort(-1);hashes.append(hashlib.sha256(perm.astype(np.uint8).tobytes()).hexdigest())
        residual=np.take_along_axis(res,perm,-1);scores=geom+residual;e=np.exp(scores-scores.max(-1,keepdims=True));r=e/e.sum(-1,keepdims=True);w=.05+.25*r
        # If the registered scale guard disables a model, match exact G1 there.
        disabled=np.all(res==0,axis=-1);w[disabled]=base_w[disabled];r[disabled]=z['r'][1][disabled]
        randomw=np.take_along_axis(base_w,perm,-1)
        for control,weights in enumerate([w,randomw]):
            probs=(member*weights[...,None]).sum(-2)
            if control==0:probs[disabled]=z['probs'][1][disabled]
            correct=probs.argmax(-1)==lab['y'];AA[control,:,repeat]=correct.mean(-1)
            for g in range(4):ga[control,:,repeat,g]=correct[:,lab['group']==g].mean(-1)
            WGA[control,:,repeat]=ga[control,:,repeat].min(-1);concentration[control,:,repeat]=(weights**2).sum(-1).mean(-1)
            rr=r if control==0 else (weights-.05)/.25
            signed=(rr*t).sum(-1);gain[control,:,repeat]=np.array([signed[s,wrong[s]].mean() for s in range(3)])
    p=ROOT/f'metrics/C_permutation/batch{batch:03d}.npz';save_npz(p,AA=AA,WGA=WGA,concentration=concentration,gain_Uwrong=gain,group_accuracy=ga,role=np.array(ROLE))
    q=draw_receipt('C_permutation',batch,before,rng,hashes)
    return dict(batch=batch,repeats=50,controls=['RH_residual_slot_permutation','G1_weight_slot_permutation']),[p,q]


def C_tables():
    z=load(ROOT/'metrics/WEIGHT_AND_SCORE_DECOMPOSITION.npz');labels=load(ROOT/'metrics/C_EVALUATION_LABELS.npz');stats=[];groups=[];mechanism=[];paths=[]
    for s in range(3):
        for mi,m in enumerate(METHODS):
            r=json.loads((ROOT/f'metrics/C_seed/seed{s}/{m}.json').read_text());stats.append(r)
            for f in range(3):
                out=load(pred_path(s,f,4,m));data=C(s,f);lab=C(s,f,True);np.testing.assert_array_equal(data['sample_id'],lab['sample_id'])
                model=json.loads((folder(s,f,4)/'model.json').read_text());g1=load(pred_path(s,f,4,'G1_frozen'))
                mechanism+=diagnostics(s,f,4,m,out,data['probs'],lab['y'],lab['sample_id'],model,g1)
            for g in range(4):
                mask=labels['group']==g;correct=z['probs'][mi,s,mask].argmax(-1)==labels['y'][mask]
                row=dict(seed=s,method=m,group=g,n=int(mask.sum()),correct=int(correct.sum()),accuracy=float(correct.mean()),AA=r['AA'],WGA=r['WGA'],
                    WCA=r['WCA'],worst_group_identity=r['worst_groups'],role=ROLE)
                for ri,ref in [(0,'U0'),(1,'G1')]:
                    old=z['probs'][ri,s,mask].argmax(-1)==labels['y'][mask]
                    row.update({f'corrected_{ref}':int((correct&~old).sum()),f'broken_{ref}':int((~correct&old).sum()),f'delta_accuracy_{ref}_pp':100*float(correct.mean()-old.mean())})
                groups.append(row)
    cirows=[];boot={k:np.concatenate([load(ROOT/f'metrics/C_bootstrap/batch{i:03d}.npz',[k])[k] for i in range(20)],axis=-1) for k in ['AA','WGA']}
    for mi,m in enumerate(METHODS):
        for ri,reference in [(0,'U0'),(1,'G1_frozen')]:
            for s in [0,1,2,-1]:
                row=dict(method=m,reference=reference,seed=s,role=ROLE)
                for k,array in boot.items():
                    d=array[mi]-array[ri];d=d.mean(0) if s==-1 else d[s]
                    row['delta_'+k+'_CI95_pp']=np.quantile(100*d,[.025,.975]).tolist()
                cirows.append(row)
    p=ROOT/'metrics/BOOTSTRAP_CI.json';jsonwrite(p,dict(role=ROLE,repeats=2000,unit='1199 distinct image IDs, shared across all methods/seeds',
        limitation='Conditional fixed pipelines, one backbone; does not quantify independent cohort or backbone variability',comparisons=cirows));paths.append(p)
    means=[];lookup={(r['seed'],r['method']):r for r in stats}
    for m in METHODS:
        ss=[lookup[s,m] for s in range(3)];row=dict(method=m,role=ROLE,n_unique_images=1199,seeds=3)
        for k in ['AA','WGA','WCA','BA','NLL','worst_class_NLL']:
            v=np.array([r[k] for r in ss]);row[k+'_mean']=float(v.mean());row[k+'_seed_SD']=float(v.std(ddof=1));row[k+'_seed_min']=float(v.min());row[k+'_seed_max']=float(v.max())
        for ref in ['U0','G1_frozen']:
            d=[100*(lookup[s,m]['AA']-lookup[s,ref]['AA']) for s in range(3)];row['delta_AA_'+ref+'_pp_by_seed']=d
            row['AA_floor_'+ref+'_every_seed']=all(v> -1 for v in d)
            for k in ['AA','WGA']:row['delta_'+k+'_'+ref+'_mean_pp']=float(np.mean([100*(lookup[s,m][k]-lookup[s,ref][k]) for s in range(3)]))
        means.append(row)
    p=ROOT/'metrics/MEAN_AA_WGA_COMPARISON.csv';csvwrite(p,means);paths.append(p)
    p=ROOT/'metrics/PER_SEED_AND_GROUP_METRICS.csv';csvwrite(p,groups);paths.append(p)
    p=ROOT/'metrics/C_PER_SEED_METRICS.csv';csvwrite(p,stats);paths.append(p)
    p=ROOT/'metrics/C_SCORE_MECHANISM.csv';csvwrite(p,mechanism);paths.append(p)
    flips=[];pred=z['probs'].argmax(-1);correct=pred==labels['y']
    for s in range(3):
        for i,sid in enumerate(z['sample_id']):
            row=dict(seed=s,fold=int(z['fold'][s,i]),sample_id=str(sid),y=int(labels['y'][i]),group=int(labels['group'][i]),role=ROLE)
            for mi,m in enumerate(METHODS):
                row.update({m+'_prediction':int(pred[mi,s,i]),m+'_correct':bool(correct[mi,s,i]),m+'_p0':float(z['probs'][mi,s,i,0]),m+'_p1':float(z['probs'][mi,s,i,1])})
                for ri,ref in [(0,'U0'),(1,'G1')]:
                    row.update({m+'_corrected_'+ref:bool(correct[mi,s,i] and not correct[ri,s,i]),m+'_broken_'+ref:bool(not correct[mi,s,i] and correct[ri,s,i]),
                        m+'_delta_p1_'+ref:float(z['probs'][mi,s,i,1]-z['probs'][ri,s,i,1])})
            flips.append(row)
    p=ROOT/'metrics/FLIP_AUDIT.csv';csvwrite(p,flips);paths.append(p)
    permutations={k:np.concatenate([load(ROOT/f'metrics/C_permutation/batch{i:03d}.npz',[k])[k] for i in range(60)],axis=2) for k in ['AA','WGA','concentration','gain_Uwrong']}
    controls=[];observed_AA=np.array([lookup[s,'G1_RH']['AA'] for s in range(3)]);observed_WGA=np.array([lookup[s,'G1_RH']['WGA'] for s in range(3)])
    for control,name in enumerate(['RH_residual_slot_permutation','G1_weight_slot_permutation']):
        aa=permutations['AA'][control];wga=permutations['WGA'][control]
        controls.append(dict(control=name,repeats=3000,AA_mean=float(aa.mean()),WGA_mean=float(wga.mean()),
            mean_AA_quantiles=np.quantile(aa.mean(0),[.025,.5,.975]).tolist(),mean_WGA_quantiles=np.quantile(wga.mean(0),[.025,.5,.975]).tolist(),
            mean_WGA_at_least_observed_RH_fraction=float((wga.mean(0)>=observed_WGA.mean()-1e-15).mean()),
            mean_WGA_strictly_above_observed_RH_fraction=float((wga.mean(0)>observed_WGA.mean()+1e-15).mean()),
            mean_WGA_tie_observed_RH_fraction=float(np.isclose(wga.mean(0),observed_WGA.mean(),atol=1e-15,rtol=0).mean()),
            per_seed_WGA_at_least_RH_fraction=(wga>=observed_WGA[:,None]-1e-15).mean(1).tolist(),
            mean_concentration=float(permutations['concentration'][control].mean()),mean_gain_Uwrong=float(permutations['gain_Uwrong'][control].mean()),role=ROLE))
    p=ROOT/'metrics/PERMUTATION_CONTROLS.json';jsonwrite(p,dict(role=ROLE,controls=controls,interpretation='Descriptive nulls only; no independent significance or selection. Residual multisets preserve strength but final mixture concentration can change; G1 weight permutation preserves final concentration.',observed_RH_AA=float(observed_AA.mean()),observed_RH_WGA=float(observed_WGA.mean())));paths.append(p)
    p=ROOT/'metrics/HISTORICAL_V6_CONTEXT.csv';historical=pd.read_csv(V6/'exploratory_c/EXPLORATORY_C_RESULTS.csv');historical=historical[historical.level=='mean_of_seeds'];csvwrite(p,historical.replace({np.nan:None}).to_dict('records'));paths.append(p)
    return dict(methods=5,seed_metrics=15,groups=60,all_image_flip_audit_rows=len(flips),bootstrap=2000,permutation=3000),paths


def oracle():
    z=load(ROOT/'metrics/WEIGHT_AND_SCORE_DECOMPOSITION.npz');lab=load(ROOT/'metrics/C_EVALUATION_LABELS.npz');rows=[];images=[]
    for s in range(3):
        y=lab['y'];u=z['probs'][0,s];member=z['member_probs'][s];true=member[np.arange(len(y))[:,None],np.arange(15)[None,:],y[:,None]]
        best=true.argmax(-1);oracle_probs=.75*u+.25*member[np.arange(len(y)),best];feasible=oracle_probs.argmax(-1)==y
        mask=(u.argmax(-1)!=y)&(lab['group']==2)
        # Retrospective intervention on group-2 Uniform-wrong images only;
        # other predictions stay G1 so the worst-group switch is visible.
        hybrid=z['probs'][1,s].copy();hybrid[mask]=oracle_probs[mask]
        row=dict(seed=s,uniform_wrong_group2=int(mask.sum()),oracle_correctable=int((feasible&mask).sum()),
            oracle_not_correctable=int((~feasible&mask).sum()),G1_plus_label_oracle_group2=metric(hybrid,y,lab['group']),
            all_image_label_oracle=metric(oracle_probs,y,lab['group']),label_aware_undeployable=True,role=ROLE)
        for mi,m in enumerate(METHODS):row[m+'_correct_on_Uniform_wrong_group2']=int(((z['probs'][mi,s].argmax(-1)==y)&mask).sum())
        rows.append(row)
        for i in np.flatnonzero(mask):images.append(dict(seed=s,sample_id=str(z['sample_id'][i]),oracle_feasible=bool(feasible[i]),best_expert=int(best[i]),oracle_true_probability=float(oracle_probs[i,y[i]]),label_aware_undeployable=True,role=ROLE))
    p=ROOT/'metrics/ORACLE_FEASIBILITY.json';jsonwrite(p,dict(role=ROLE,seeds=rows,unique_Uniform_wrong_group2_images=len(set(r['sample_id'] for r in images)),
        limitation='Uses C labels/group2 retrospectively. Not a learned model, deployable candidate, selection criterion or future WGA target.'))
    q=ROOT/'metrics/ORACLE_FEASIBILITY_IMAGES.csv';csvwrite(q,images);return dict(label_aware_undeployable=True,correctable_rows=sum(r['oracle_correctable'] for r in rows)),[p,q]


def report():
    mean=pd.read_csv(ROOT/'metrics/MEAN_AA_WGA_COMPARISON.csv');cs=pd.read_csv(ROOT/'metrics/C_PER_SEED_METRICS.csv');bs=pd.read_csv(ROOT/'metrics/B_SEED_METRICS.csv');b=pd.read_csv(ROOT/'metrics/B_OOF_MECHANISM.csv')
    gate=json.loads((ROOT/'metrics/B_DIRECTION_DECISION.json').read_text());cis=json.loads((ROOT/'metrics/BOOTSTRAP_CI.json').read_text());perm=json.loads((ROOT/'metrics/PERMUTATION_CONTROLS.json').read_text());orc=json.loads((ROOT/'metrics/ORACLE_FEASIBILITY.json').read_text())
    head=pd.read_csv(ROOT/'metrics/B_HEAD_EFFECTS.csv');stability=pd.read_csv(ROOT/'metrics/B_HEAD_EFFECT_STABILITY.csv');margin=pd.read_csv(ROOT/'metrics/B_MARGIN_DIAGNOSTICS.csv');cm=pd.read_csv(ROOT/'metrics/C_SCORE_MECHANISM.csv')
    primary=mean[mean.method=='G1_RH'].iloc[0];d=float(primary.delta_WGA_G1_frozen_mean_pp);verdict='EXPLORATORY_C_WGA_UP' if d>1e-12 else 'EXPLORATORY_C_WGA_DOWN' if d< -1e-12 else 'EXPLORATORY_C_WGA_EQUAL'
    full_models=[json.loads((folder(s,f,4)/'model.json').read_text()) for s in range(3) for f in range(3)];inner_models=[json.loads((folder(s,f,i)/'model.json').read_text()) for s in range(3) for f in range(3) for i in range(4)]
    scales=[dict(seed=s,fold=f,part=i,kind=kind,**json.loads((folder(s,f,i)/'model.json').read_text())['scales'][kind],role=ROLE) for s in range(3) for f in range(3) for i in range(5) for kind in ['pooled','head']]
    scale_path=ROOT/'metrics/FITTED_SCALE_SUMMARY.csv';csvwrite(scale_path,scales)
    lines=['# G1 residual v7: continuous precision weighting results','', '**'+ROLE+'**','', '**INDEPENDENT_CONFIRMATION_NOT_PERFORMED**','',
        f"Frozen primary: **G1_RH**. B verdict: **{gate['status']}** ({gate['uncertainty']}). C verdict versus G1: **{verdict}**. No method is promoted. Original v6 NO_GO and Uniform/G1 production inference remain unchanged.",'',
        'The study implements the requested optimization, not just baseline confirmation: six-column continuous signed-delta ridge (explicit penalty 10), exact-head/predicted-class shrunk residuals (kappa 25, clip 0.1), nested predicted-OOF scale matching, and the separately registered margin-gated RHM ablation. Raw G1 geometry, gamma 50, alpha 0.25, every prototype head, backbone, member prediction and IPS training are unchanged. Lambda 0.5 is fixed; no lambda1/grid or C-driven selection was run.','',
        'Nine matching ensembles have 240 B images each. Four B crossfit partitions use seed 71000+100*seed+fold; each model uses three secondary class-stratified OOF partitions entirely inside its owning B-training set (seed 72000+1000*seed+100*fold+10*part). There are 36 B-OOF model fits plus nine full-B fits, and 135 secondary ridge/head fits. Every feature RMS, per-head mu/sigma, q20, head effect and residual scale is B-training-only. Pooled and head-aware residual scales are estimated separately. The only numerical signal threshold is the preregistered s_R<0.002 guard. Finite/support failures also block unsafe fits; no post-hoc OOF quality cutoff was introduced.','',
        'B targets use (2y−1)*(p_j(1)−raw_Uniform(1)); raw float32 Uniform reduction is retained for exact baselines. Target/true-class-shift identities hold within float32 roundoff. No group field enters B fitting or evaluation. Repeated B images are collapsed by seed before seed metrics: probabilities and scalar gain/ranking diagnostics are averaged over their owning ensembles, never head-slot coefficients across ensembles.','',
        '## B-OOF expert-selection evidence','', '| Method | Seed | Unique images | Uniform-wrong | Mean gain on Uniform-wrong | Δgain vs G1 | Pair rank on Uniform-wrong | AA (%) | ΔAA vs U0 (pp) | ΔAA vs G1 (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in bs.itertuples():lines.append(f'| {r.method} | {r.seed} | {r.unique_images} | {r.uniform_wrong_images} | {r.mean_gain_Uwrong:+.7f} | {r.gain_difference_Uwrong:+.7f} | {r.pair_ranking_Uwrong:.5f} | {100*r.AA:.6f} | {r.delta_AA_U0_pp:+.6f} | {r.delta_AA_G1_pp:+.6f} |')
    lines+=['',f"1. **Continuous ranking/direction:** RH mean signed-gain difference on seed-pooled Uniform-wrong B images is {gate['mean_gain_difference_Uwrong']:+.8f}; per seed {gate['per_seed_gain_difference']}; {gate['positive_seeds']}/3 positive. Shared-ID paired 95% CI {gate['gain_CI95']}. Direction gate passes: {gate['direction_pass']}. Distinct Uniform-wrong B counts per seed are in the table; they are not independent cohorts. Per-image ranks (with valid-count handling), geometry comparison and all gains are in B_OOF_MECHANISM.csv and B_OOF_SIGNED_GAIN.csv.",
        '', f"2. **Head effects versus pooled:** head-aware B gain differences must be assessed against R in the table, not inferred from coefficient size. Across {len(head)} fitted head/class effects, median effective image support {head.effective_images.median():.3f}; {(head.distinct_images<30).sum()} have <30 distinct images; {(head.b>0).sum()} positive and {(head.b<0).sum()} negative. Within the same actual ensemble, mean four-fold sign agreement is {stability.sign_agreement.mean():.4f}; median b SD {stability.std_b.median():.6f}. Full head SHA/support/denominator/clipping/shrinkage and same-ensemble stability are retained. A stable slot is not a causal specialist; no head identity is transferred across folds.",
        '', '3. **Scale and actual score influence:**','', '| Method | Split | Mean weight TV vs G1 | Top-head changes | Mean residual/geometry centered RMS | Mean clipping fraction |', '|---|---|---:|---:|---:|---:|']
    for split,frame in [('B-OOF',b),('C exploratory',cm)]:
        for m in METHODS[2:]:
            r=frame[frame.method==m];lines.append(f'| {m} | {split} | {r.weight_TV.mean():.8f} | {int(r.top_changed.sum())}/{len(r)} | {r.score_RMS_ratio.mean():.6f} | {r.clip_fraction.mean():.6f} |')
    lines+=['', f"Enabled scales: B-inner pooled {sum(m['scales']['pooled']['enabled'] for m in inner_models)}/36, head {sum(m['scales']['head']['enabled'] for m in inner_models)}/36; full-B pooled {sum(m['scales']['pooled']['enabled'] for m in full_models)}/9, head {sum(m['scales']['head']['enabled'] for m in full_models)}/9. All measured s_R/s_G, OOF-vs-null MSE and off reasons are in FITTED_SCALE_SUMMARY.csv. An off-switch is an explicit registered G1 fallback, not an accidental implementation alias.",
        '', '4. **Direction versus actual threshold crossings:** seed-pooled corrected/broken counts versus both references are in B_SEED_METRICS.csv. The following are distinct per-seed image counts, not expert rows:', '', '| Seed | Method | Corrected U0 | Broken U0 | Corrected G1 | Broken G1 | G1-negative → positive gain on Uniform-wrong |', '|---:|---|---:|---:|---:|---:|---:|']
    for r in bs.itertuples():
        if r.method in METHODS[2:]:lines.append(f'| {r.seed} | {r.method} | {r.corrected_U0} | {r.broken_U0} | {r.corrected_G1} | {r.broken_G1} | {r.G1_negative_to_positive_Uwrong} |')
    lines+=['', '5. **Margin gate:** this is a fixed ablation, never a replacement primary selected by C. It is not an AA guarantee. Strata use each B-training q20/q80. Counts below are B validation rows with repeated IDs across outer ensembles; seed-pooled accuracy is reported separately.', '', '| Method | Margin stratum | Rows | Mean gate | Corrected U0 | Broken U0 | Corrected G1 | Broken G1 |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in margin.itertuples():
        if r.method in ['G1_RH','G1_RHM']:lines.append(f'| {r.method} | {r.margin_stratum} | {r.n} | {r.mean_gate:.6f} | {r.corrected_U0} | {r.broken_U0} | {r.corrected_G1} | {r.broken_G1} |')
    lines+=['', '## C exploratory downstream evidence','', '6. **All locked methods, all outcomes:** 1,199 unique images per seed (same IDs across three seeds; 3,597 seed-image rows). Group counts per seed are [467,466,133,133]. One shared frozen ERM* backbone; this is not an ERM-only run or independent test. WGA is computed as each seed minimum first, then averaged.', '',
        '| Method | Mean AA (%) | Mean WGA (%) | Seed SD WGA (pp) | ΔAA vs U0 (pp) | ΔWGA vs U0 (pp) | ΔWGA vs G1 (pp) |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in mean.itertuples():lines.append(f'| {r.method} | {100*r.AA_mean:.6f} | {100*r.WGA_mean:.6f} | {100*r.WGA_seed_SD:.6f} | {r.delta_AA_U0_mean_pp:+.6f} | {r.delta_WGA_U0_mean_pp:+.6f} | {r.delta_WGA_G1_frozen_mean_pp:+.6f} |')
    lines+=['', '| Seed | Method | AA (%) | WGA (%) | Worst class acc (%) | Worst group(s) |', '|---:|---|---:|---:|---:|---|']
    for r in cs.itertuples():lines.append(f'| {r.seed} | {r.method} | {100*r.AA:.6f} | {100*r.WGA:.6f} | {100*r.WCA:.6f} | {r.worst_groups} |')
    lines+=['', '7. **Uncertainty and nulls:** 2,000 group-stratified unique-image draws share image IDs across all seeds/methods; intervals are conditional on these fixed pipelines. No independent-row bootstrap over 3,597 repeated entries. B bootstrap likewise shares union-image draws and uses seed-specific availability/subset masks. These intervals do not estimate new-cohort/backbone uncertainty.', '', '| Method | Reference | Mean ΔAA 95% CI (pp) | Mean ΔWGA 95% CI (pp) |', '|---|---|---|---|']
    for r in cis['comparisons']:
        if r['seed']==-1:lines.append(f"| {r['method']} | {r['reference']} | {r['delta_AA_CI95_pp']} | {r['delta_WGA_CI95_pp']} |")
    for r in perm['controls']:lines+=['',f"Control {r['control']}: 3000 fixed-seed repeats; mean WGA {100*r['WGA_mean']:.6f}%; fraction at least observed RH WGA {r['mean_WGA_at_least_observed_RH_fraction']:.6f}, strictly above {r['mean_WGA_strictly_above_observed_RH_fraction']:.6f}, ties {r['mean_WGA_tie_observed_RH_fraction']:.6f}. These are descriptive null frequencies, not independent significance tests. Residual permutations preserve each image's residual multiset/strength but can change final concentration; G1 weight permutations preserve final concentration."]
    lines+=['', '8. **Strict AA floor:** B primary passes every seed versus U0: '+str(gate['AA_floor_every_seed'])+'; differences '+str(gate['per_seed_delta_AA_U0_pp'])+' pp. C eligibility for every preregistered method:', '', '| Method | Per-seed ΔAA vs U0 (pp) | Every seed > −1 pp | Per-seed ΔAA vs G1 (pp) |', '|---|---|---|---|']
    for r in mean.itertuples():lines.append(f'| {r.method} | {r.delta_AA_U0_pp_by_seed} | {r.AA_floor_U0_every_seed} | {r.delta_AA_G1_frozen_pp_by_seed} |')
    lines+=['', '9. **Failure/success decomposition:**',
        f"B proxy RH−G1 worst-class NLL = {gate['worst_class_NLL_delta']:+.8f}; tolerance 0.003 passes: {gate['proxy_pass']}. Direction passes: {gate['direction_pass']}; AA passes: {gate['AA_floor_every_seed']}. Thus the exact B decision is {gate['status']}. Poor/uncertain OOF advantage ranking, guard-disabled scale, head-support instability, and actual correction/breakage counts are separate diagnostics above. Coefficient/weight movement alone does not establish expert-selection improvement. Limited 180-image B-inner-training support is a limitation, not a proven sole cause. No guard threshold or residual strength was changed after observing outcomes.",
        '', f"Retrospective oracle: {sum(r['oracle_correctable'] for r in orc['seeds'])}/{sum(r['uniform_wrong_group2'] for r in orc['seeds'])} Uniform-wrong group-2 seed-image rows are correctable at fixed alpha using label-aware best-head routing; {orc['unique_Uniform_wrong_group2_images']} distinct hard image IDs are involved. Per-seed counts, current-method corrections and whether another group becomes worst are in ORACLE_FEASIBILITY.json. Oracle labels never enter fitting, selection or deployed inference; this is feasibility, not learnability or a future WGA target.",
        '', '10. **Persistence and unchanged production:** every secondary fit, B model, validation inference/evaluation, full-B fit, C inference/evaluation and uncertainty batch has an atomic completion receipt with source/config/input/head/split fingerprints. Real workers are force-killed after a committed secondary fit and after a committed C inference; recovered artifacts must match uninterrupted byte hashes and completed mtimes. Validated acceptance outputs are reused. CHECKSUMS.sha256 and DELIVERY_VALIDATION.json verify numerical invariants and protected v1-v6 artifacts. state.json, HANDOFF_STATUS.md and NEXT_AGENT_PROMPT.md are refreshed after every task. A completed --resume --verify reports the exact count; a separate-process zero-work resume is recorded at delivery.',
        '', '## Limitations and continuation','',
        'C is reused development, with original official labels and the documented species-label caveat. No silent relabeling. Shared backbone/head seeds are not independent cohorts. B image support and Uniform-wrong counts are small; class/boundary weighting and in-sample head-residual fitting can be optimistic, hence nested B-OOF evaluation. A common-offset/zero-delta/scale guard has automated coverage. Inference has no y/group/ID lookup. All negative ablations are published. Historical v6 results are loaded as context only; no G2/CCW rerun occurred.',
        '', '**NO_FRESH_DATA — independent confirmation is blocked.** Do not rename/repartition old C as fresh. If B is NO_GO, a future fresh primary comparison remains G1 versus Uniform; v7 stays diagnostic. If B passes, only the locked RH recipe could be preregistered for a truly independent cohort. Nothing here authorizes integration or C-driven tuning.',
        '', '```powershell',COMMAND,'.venv/Scripts/python.exe -m experiments.g1_residual_v7.run --verify','.venv/Scripts/python.exe -m pytest experiments/g1_residual_v7/tests -q','```','',
        'Required files are under metrics/: MEAN_AA_WGA_COMPARISON.csv, PER_SEED_AND_GROUP_METRICS.csv, B_OOF_MECHANISM.csv, B_OOF_SIGNED_GAIN.csv, WEIGHT_AND_SCORE_DECOMPOSITION.npz (label-free), C_EVALUATION_LABELS.npz (evaluator-only), FLIP_AUDIT.csv (every image, including unchanged), PERMUTATION_CONTROLS.json and BOOTSTRAP_CI.json. Exact fitted models, nested OOF estimates and per-fold outputs remain under checkpoints/.','']
    p=ROOT/'EXPLORATORY_C_REPORT.md';atomic(p,'\n'.join(lines));q=ROOT/'CONFIRMATION_BLOCKED.md';atomic(q,'# NO_FRESH_DATA\n\nINDEPENDENT_CONFIRMATION_NOT_PERFORMED\n\nOnly previously inspected C images and a shared ERM* backbone are authorized/available for this study. No production promotion. Preserve original v6 NO_GO.\n')
    limitations=ROOT/'LIMITATIONS_AND_NEXT_STEPS.md';atomic(limitations,'# v7 limitations and next steps\n\n'+ROLE+'\n\n'+gate['status']+'; '+verdict+'; INDEPENDENT_CONFIRMATION_NOT_PERFORMED.\n\n'
        'No hyperparameter search follows these C outcomes. Read the quantified mechanism report before proposing a separately preregistered intervention. '
        'Fixed original heads/backbone, small B support, repeated IDs, conditional uncertainty and original-label caveat limit interpretation. '
        'No claim that an oracle gap is learnable or that all weighting is at a ceiling. '
        'Fresh confirmation needs verifiably untouched images and/or independent backbone replication, with matched fixed heads and a B-only locked candidate.\n')
    verdict_path=ROOT/'metrics/FINAL_VERDICT.json';jsonwrite(verdict_path,dict(role=ROLE,B_verdict=gate['status'],B_uncertainty=gate['uncertainty'],C_verdict=verdict,primary='G1_RH',independent_confirmation='INDEPENDENT_CONFIRMATION_NOT_PERFORMED',promotion=False))
    return dict(B_verdict=gate['status'],C_verdict=verdict,primary='G1_RH'),[p,q,limitations,verdict_path,scale_path]


def complete():
    from .model import features,target
    from .run import train_indices,secondary_splits,manifest
    audited=0
    for s in range(3):
        for f in range(3):
            b=B(s,f)
            for part in range(5):
                owning=train_indices(b,s,f,part);final=json.loads((folder(s,f,part)/'model.json').read_text());fits=[(final,owning)]
                for k,(a,v) in enumerate(secondary_splits(b,s,f,part)):
                    sub=json.loads((folder(s,f,part)/f'scale_subcv/sub{k}/model.json').read_text());assert not set(sub['fit_ids'])&set(b['sample_id'][v]);fits.append((sub,a))
                    assert set(sub['fit_ids'])<=set(b['sample_id'][owning])
                for model,idx in fits:
                    assert model['fit_ids']==b['sample_id'][idx].tolist() and model['head_sha256']==b['head_sha256'].tolist()
                    dm=b['distances'][idx].astype(float).min(-1);np.testing.assert_array_equal(model['normalizer']['mu'],dm.mean(0));np.testing.assert_array_equal(model['normalizer']['sigma'],dm.std(0))
                    X,qq,zz,delta,u,pred=features(b['probs'][idx],b['distances'][idx],model['normalizer']);np.testing.assert_allclose(model['column_scale'],np.maximum(np.sqrt((X*X).mean((0,1))),1e-3),atol=1e-15,rtol=0)
                    assert model['q20']==float(np.quantile(abs(u[:,1]-.5),.2));y=b['y'][idx];counts=np.bincount(y,minlength=2)
                    omega=len(idx)/(2*counts[y])*(1+(abs(u[:,1]-.5)<=model['q20']));omega/=omega.mean();xx=(X/np.array(model['column_scale'])).reshape(-1,6);w=np.repeat(omega/15,15);t=target(b['probs'][idx],y)
                    normal=xx.T@(w[:,None]*xx)+10*np.eye(6);rhs=xx.T@(w*t.ravel());np.testing.assert_allclose(normal@np.array(model['theta']),rhs,atol=1e-11,rtol=1e-12)
                    pooled=(xx@np.array(model['theta'])).reshape(len(idx),15);e=t-pooled
                    for cell in model['cells']:
                        j=cell['head'];mask=pred[:,j]==cell['predicted_class'];n=float(omega[mask].sum());den=float((omega[mask]*qq[mask,j]**2).sum());num=float((omega[mask]*qq[mask,j]*e[mask,j]).sum());expected=float(np.clip(num/(den+1e-8),-.1,.1)*n/(n+25)) if n else 0.
                        assert abs(expected-cell['b'])<1e-14 and abs(n-cell['effective_images'])<1e-12
                    audited+=1
                oof=load(folder(s,f,part)/'B_scale_OOF.npz');np.testing.assert_array_equal(oof['sample_id'],b['sample_id'][owning])
                for kind in ['pooled','head']:np.testing.assert_allclose(final['scales'][kind]['s_R'],np.sqrt(np.mean(center(oof[kind])**2)),atol=1e-15,rtol=0)
                if part<4:assert not set(final['fit_ids'])&set(b['sample_id'][outer_splits(b,s,f)[part][1]])
                else:assert not set(final['fit_ids'])&set(C(s,f)['sample_id'])
    decomp=load(ROOT/'metrics/WEIGHT_AND_SCORE_DECOMPOSITION.npz');assert not {'y','group'}&decomp.keys()
    assert decomp['weights'].shape==(5,3,1199,15) and decomp['probs'].shape==(5,3,1199,2)
    np.testing.assert_allclose(decomp['weights'].sum(-1),1,atol=2e-6,rtol=0);assert (decomp['weights']>=.05-1e-8).all()
    records=[json.loads(p.read_text()) for p in (ROOT/'completed').glob('*.json')]
    infer_times=[r['completed_utc'] for r in records if r['id'].startswith('C_infer:')];eval_times=[r['completed_utc'] for r in records if r['id'].startswith('C_eval:')];assert max(infer_times)<min(eval_times)
    for r in records:
        for name,a in r['artifacts'].items():assert sha(ROOT/name)==a['sha256']
    validation=ROOT/'DELIVERY_VALIDATION.json';jsonwrite(validation,dict(role=ROLE,status='PASS',ridge_and_head_fits_independently_audited=audited,
        train_only_stats_and_normal_equations=True,all_nested_scale_OOF_ownership_verified=True,C_predictions_before_evaluation=True,
        all_existing_receipt_artifacts_verified=True,receipt_count_before_final=len(records),unchanged_production=True))
    paths=sorted(set(ROOT/name for r in records for name in r['artifacts'])|{validation,ROOT/'EXPERIMENT_LOCK.json',ROOT/'config.lock.json',ROOT/'manifest.json'})
    checksums=ROOT/'CHECKSUMS.sha256';atomic(checksums,'\n'.join(sha(p)+'  '+p.relative_to(ROOT).as_posix() for p in paths)+'\n')
    return dict(status='DONE_EXPLORATORY',audited_fits=audited,receipt_count_before_final=len(records)),[validation,checksums]
