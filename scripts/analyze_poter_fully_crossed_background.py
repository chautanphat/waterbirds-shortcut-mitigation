"""Analyze actual frozen POTER predictions using species and scene as crossed units."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from run_poter_fully_crossed_background import CONFIG, OUTPUT, require
from src.counterfactual.common import sha256, write_json

LABELS = ("Landbird", "Waterbird")
METRICS = ("margin_D", "delta_p", "water_minus_land_error", "cross_type_flip_rate", "within_type_scene_margin_sd")


def load(output):
    fg = pd.read_csv(output / "foreground_manifest.csv")
    bg = pd.read_csv(output / "background_manifest.csv")
    pred = pd.read_csv(output / "fully_crossed_predictions.csv", float_precision="round_trip")
    stamp = json.loads((output / "run_lock.json").read_text(encoding="utf-8"))
    require(stamp["foreground_manifest_sha256"] == sha256(output / "foreground_manifest.csv") and
            stamp["background_manifest_sha256"] == sha256(output / "background_manifest.csv"), "Run manifests differ")
    require(len(pred) == 6400 and len(fg) == fg.source_image_id.nunique() == 160 and
            len(bg) == bg.background_id.nunique() == 40 and set(pred.model_id) == {"POTER"}, "Design size differs")
    require(not pred.duplicated(["source_image_id","background_id"]).any(), "Duplicate grid keys")
    indexed = pred.set_index(["source_image_id", "background_id"])
    expected = pd.MultiIndex.from_product([fg.source_image_id, bg.background_id], names=indexed.index.names)
    require(set(indexed.index) == set(expected), "Crossing is incomplete")
    pred = indexed.reindex(expected).reset_index()
    for name in ("alpha_sha256", "foreground_layer_sha256", "mask_sha256", "true_y", "species_id", "original_group"):
        require(pred[name].eq(pred.source_image_id.map(fg.set_index("source_image_id")[name])).all(), f"Foreground drift in {name}")
    for name in ("background_type", "background_bank", "background_sha256"):
        require(pred[name].eq(pred.background_id.map(bg.set_index("background_id")[name])).all(), f"Scene drift in {name}")
    require(pred.checkpoint_sha256.eq(stamp["model"]["checkpoint_sha256"]).all(), "Mixed checkpoints")
    require(pred.groupby("source_image_id").model_input_sha256.nunique().eq(40).all(), "Duplicate model-view rasters")
    require(np.isfinite(pred[["margin_water_minus_land", "p_waterbird"]]).all().all() and pred.p_waterbird.between(0,1).all()
            and np.allclose(pred.margin_water_minus_land, pred.logit_waterbird-pred.logit_landbird)
            and np.allclose(pred.p_waterbird, 1/(1+np.exp(-pred.margin_water_minus_land)), atol=1e-7)
            and pred.pred_y.eq(pred.margin_water_minus_land.gt(0).astype(int)).all()
            and pred.correct.eq(pred.pred_y.eq(pred.true_y)).all(), "Frozen predictions or class formula differs")
    arrays = {dest: pred[col].to_numpy().reshape(len(fg), len(bg)) for dest,col in {
        "margin":"margin_water_minus_land","p":"p_waterbird","error":"correct", "pred":"pred_y"}.items()}
    arrays["error"] = 1. - arrays["error"].astype(float)
    return fg, bg, pred, arrays


def scene_indices(bg, bank):
    scope = bg.background_bank.eq(bank) if bank != "ALL" else np.ones(len(bg), dtype=bool)
    return tuple(np.flatnonzero(scope & bg.background_type.eq(kind)) for kind in ("land", "water"))


def foreground_effects(fg, bg, arrays):
    results = []
    for bank in ("A","B","ALL"):
        land, water = scene_indices(bg, bank)
        p, margin, error, preds = (arrays[key] for key in ("p", "margin", "error", "pred"))
        row = fg[["source_image_id", "species_id", "true_y", "original_group", "mask_area_fraction", "soft_boundary_fraction"]].copy()
        row["model_id"], row["bank"] = "POTER", bank
        row["D"] = margin[:,water].mean(1)-margin[:,land].mean(1)
        row["DeltaP"] = p[:,water].mean(1)-p[:,land].mean(1)
        for kind, indices in (("land",land),("water",water)):
            row[f"{kind}_error_rate"] = error[:,indices].mean(1)
            q = preds[:,indices].mean(1)
            row[f"within_{kind}_flip_rate"] = 2*q*(1-q)*len(indices)/(len(indices)-1)
        l, w = preds[:,land].mean(1), preds[:,water].mean(1)
        row["cross_type_flip_rate"] = l*(1-w)+(1-l)*w
        active = np.r_[land, water]
        row["hard_labels"] = np.ptp(preds[:,active],axis=1)+1
        row["error_rate"] = error[:,active].mean(1)
        row["margin_std"] = margin[:,active].std(1)
        row["margin_range"] = np.ptp(margin[:,active],axis=1)
        row["p_std"] = p[:,active].std(1)
        row["p_range"] = np.ptp(p[:,active],axis=1)
        le, we = row.land_error_rate, row.water_error_rate
        row["category"] = np.select([le.eq(0)&we.eq(0), le.eq(1)&we.eq(1), le.eq(.5)|we.eq(.5),
                                    le.lt(.5)&we.gt(.5), le.gt(.5)&we.lt(.5), le.gt(.5)&we.gt(.5)],
                                   ["all_correct", "all_wrong", "tie", "land_majority_correct_water_wrong",
                                    "water_majority_correct_land_wrong", "both_majority_wrong"], default="mixed_both_majority_correct")
        results.append(row)
    return pd.concat(results, ignore_index=True)


def crossed_metrics(arrays, foreground_weights, land, water, lw, ww):
    ret = {}
    for field, name in (("margin", "margin_D"), ("p", "delta_p"), ("error", "water_minus_land_error")):
        matrix = arrays[field]
        ret[name] = ((foreground_weights@matrix[:,water])*ww).sum(1)-((foreground_weights@matrix[:,land])*lw).sum(1)
    land_pred = lw @ arrays["pred"][:,land].T
    water_pred = ww @ arrays["pred"][:,water].T
    ret["cross_type_flip_rate"] = ((land_pred*(1-water_pred)+(1-land_pred)*water_pred)*foreground_weights).sum(1)
    variance = np.zeros(len(foreground_weights))
    for idx, weights in ((land,lw),(water,ww)):
        scene_scores = foreground_weights@arrays["margin"][:,idx]
        mean = (scene_scores*weights).sum(1)
        variance += ((scene_scores**2*weights).sum(1)-mean**2)/2
    ret["within_type_scene_margin_sd"] = np.sqrt(np.maximum(0.,variance))
    return ret


def bootstrap(fg, bg, arrays, settings):
    rng = np.random.default_rng(settings["bootstrap_seed"])
    nboot = settings["bootstrap_replicates"]
    records = []
    for bank in ("A", "B", "ALL"):
        land, water = scene_indices(bg, bank)
        fixed = [np.full((nboot,len(idx)),1/len(idx)) for idx in (land,water)]
        scene_draws = []
        for indices in (land, water):
            weights = np.zeros((nboot,len(indices)))
            for name in ("A","B"):
                sub = np.flatnonzero(bg.iloc[indices].background_bank.eq(name).to_numpy())
                if len(sub):
                    weights[:,sub] = rng.multinomial(len(sub), np.full(len(sub),1/len(sub)), nboot)/len(indices)
            scene_draws.append(weights)
        for mode, bg_weights in (("species_cluster_fixed_scenes",fixed),("crossed_species_scene",scene_draws)):
            data_by_class = []
            for cls in (0,1):
                keep = fg.true_y.eq(cls).to_numpy()
                species, inverse = np.unique(fg.loc[keep,"species_id"].to_numpy(), return_inverse=True)
                count = rng.multinomial(len(species),np.full(len(species),1/len(species)),nboot)[:,inverse]
                weights = count/count.sum(1, keepdims=True)
                matrix = {k:v[keep] for k,v in arrays.items()}
                point = crossed_metrics(matrix,np.full((1,keep.sum()),1/keep.sum()),land,water,
                                        fixed[0][:1],fixed[1][:1])
                simulated = crossed_metrics(matrix,weights,land,water,*bg_weights)
                data_by_class.append((point,simulated,int(keep.sum()),len(species)))
            pooled = tuple({m:(a[m]+b[m])/2 for m in METRICS} for a,b in zip(data_by_class[0][:2],data_by_class[1][:2]))
            for label, (point, simulated, n, n_species) in zip((*LABELS,"balanced"), [*data_by_class,(*pooled,len(fg),fg.species_id.nunique())]):
                for key in METRICS:
                    lower, upper = np.quantile(simulated[key],[.025,.975])
                    records.append(dict(bank=bank,true_class=label,method=mode,metric=key,estimate=float(point[key][0]),
                                        ci_low=float(lower),ci_high=float(upper),n_foregrounds=n,n_species=n_species,
                                        n_land_scenes=len(land),n_water_scenes=len(water),bootstrap_replicates=nboot,
                                        foreground_unit="species within true class",scene_unit="scene within bank/type" if mode=="crossed_species_scene" else "scenes fixed"))
    return pd.DataFrame(records)


def omissions(fg, bg, arrays):
    rows = []
    for bank in ("A","B"):
        land,water = scene_indices(bg,bank)
        for label in (0,1):
            cls = np.flatnonzero(fg.true_y.eq(label))
            tasks = [("scene",str(bg.iloc[i].background_id),cls,land[land!=i],water[water!=i]) for i in np.r_[land,water]]
            for s in fg.iloc[cls].species_id.unique():
                tasks.append(("species",s,cls[fg.iloc[cls].species_id.to_numpy()!=s],land,water))
            for unit, ident, keep, l,w in tasks:
                values = crossed_metrics({k:v[keep] for k,v in arrays.items()},np.full((1,len(keep)),1/len(keep)),l,w,
                                         np.full((1,len(l)),1/len(l)),np.full((1,len(w)),1/len(w)))
                rows.append(dict(bank=bank,true_class=LABELS[label],omitted_unit=unit,omitted_id=ident,
                                 **{k:float(v[0]) for k,v in values.items()}))
    return pd.DataFrame(rows)


def matched_effects(fg, bg, arrays, matches):
    rows = []
    for bank, selected in matches.groupby("bank"):
        l = np.flatnonzero(bg.background_id.isin(selected.land_id))
        w = np.flatnonzero(bg.background_id.isin(selected.water_id))
        for cls in (0,1):
            keep = fg.true_y.eq(cls).to_numpy()
            values = crossed_metrics({k:v[keep] for k,v in arrays.items()},np.full((1,keep.sum()),1/keep.sum()),l,w,
                                     np.full((1,len(l)),1/len(l)),np.full((1,len(w)),1/len(w)))
            rows.append(dict(bank=bank,true_class=LABELS[cls],backgrounds_per_type=len(l),
                             mean_standardized_match_distance=selected.cost.mean(), **{k:float(v[0]) for k,v in values.items()}))
    return pd.DataFrame(rows)


def grouped_outputs(output, fg, bg, pred, effects, arrays):
    data = pred.assign(loss=np.logaddexp(0.,-(2*pred.true_y-1)*pred.margin_water_minus_land))
    group = data.groupby(["model_id","background_bank","true_y","background_type"],as_index=False).agg(
        n=("correct","size"), correct=("correct","sum"), accuracy=("correct","mean"), loss=("loss","mean"),
        mean_margin=("margin_water_minus_land","mean"), n_sources=("source_image_id","nunique"), n_scenes=("background_id","nunique"))
    group.to_csv(output/"group_metrics.csv",index=False)
    scene_rows = []
    for j,item in bg.iterrows():
        bank_scene = np.flatnonzero(bg.background_bank.eq(item.background_bank))
        row = item.to_dict()
        for cls,label in enumerate(("landbird","waterbird")):
            keep = fg.true_y.eq(cls).to_numpy()
            row[f"{label}_mean_margin_centered"] = float((arrays["margin"][keep,j]-arrays["margin"][keep][:,bank_scene].mean(1)).mean())
            row[f"{label}_mean_p_centered"] = float((arrays["p"][keep,j]-arrays["p"][keep][:,bank_scene].mean(1)).mean())
            row[f"{label}_error_rate"] = float(arrays["error"][keep,j].mean())
        scene_rows.append(row)
    scenes = pd.DataFrame(scene_rows)
    scenes.to_csv(output/"background_effects.csv",index=False)
    scope = effects.groupby(["bank","true_y","original_group","species_id"],as_index=False).agg(
        n=("D","size"), mean_D=("D","mean"), mean_DeltaP=("DeltaP","mean"),
        cross_type_flip_rate=("cross_type_flip_rate","mean"), mean_error=("error_rate","mean"))
    scope.to_csv(output/"species_group_effects.csv",index=False)
    category = effects.query("true_y==0").groupby(["bank","original_group","category"],as_index=False).size()
    category["group_total"] = category.groupby(["bank","original_group"])["size"].transform("sum")
    category["share"] = category["size"]/category.group_total
    category.to_csv(output/"landbird_patterns.csv",index=False)
    qc = []
    for variable in ("mask_area_fraction","soft_boundary_fraction"):
        strata = effects.assign(quartile=pd.qcut(effects[variable].rank(method="dense"),4,labels=False)+1)
        block = strata.groupby(["bank","true_y","quartile"],as_index=False).agg(
            n=("D","size"), mean_D=("D","mean"), mean_DeltaP=("DeltaP","mean"),cross_type_flip=("cross_type_flip_rate","mean"),
            minimum=(variable,"min"),maximum=(variable,"max"))
        qc.append(block.assign(stratification=variable))
    pd.concat(qc).to_csv(output/"mask_quality_strata.csv",index=False)
    return group,scenes,category


def make_plots(output, effects, scenes, group, influence):
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    colors = ("#255e7a", "#ad5635")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(plots/name,dpi=145,bbox_inches="tight")
        plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,3.4),sharex=True)
    for ax,label in zip(axes,(0,1)):
        for bank,color in zip("AB",colors):
            x=effects.loc[(effects.bank==bank)&(effects.true_y==label),"DeltaP"]*100
            ax.hist(x,bins=np.linspace(-30,30,31),histtype="stepfilled",alpha=.28,color=color,label=f"Bank {bank}")
        ax.axvline(0,color="black",lw=.8); ax.set_title(LABELS[label]); ax.set_xlabel("Water minus Land p(Waterbird), pp"); ax.legend()
    save(fig,"signed_delta_p_histogram.png")

    fig,axes=plt.subplots(2,1,figsize=(13,6.5),sharex=True)
    order=scenes.sort_values(["background_bank","background_type","landbird_mean_p_centered"])
    colors_by_type=order.background_type.map({"land":colors[0],"water":colors[1]})
    for label,ax in zip(("landbird","waterbird"),axes):
        ax.bar(np.arange(len(order)),100*order[f"{label}_mean_p_centered"],color=colors_by_type)
        ax.axvline(19.5,color="black",lw=.6); ax.set_ylabel("Centered p(W), pp"); ax.set_title(label.title())
    axes[1].set_xticks(np.arange(len(order)),order.background_id.str.replace("train-bg-",""),rotation=75)
    save(fig,"per_scene_signed_shifts.png")

    fig,ax=plt.subplots(figsize=(6,5))
    for cls,color in enumerate(colors):
        x=effects.loc[effects.true_y.eq(cls)].pivot(index="source_image_id",columns="bank",values="DeltaP")
        ax.scatter(x.A*100,x.B*100,label=LABELS[cls],s=20,alpha=.7,color=color)
    ax.axhline(0,color="#888",lw=.6); ax.axvline(0,color="#888",lw=.6)
    ax.set_xlabel("Bank A: water minus land p(W), pp"); ax.set_ylabel("Bank B: water minus land p(W), pp"); ax.legend()
    save(fig,"bank_replication.png")

    fig,ax=plt.subplots(figsize=(9,3.8))
    g=group.sort_values(["background_bank","true_y","background_type"])
    ax.bar(np.arange(len(g)),100*(1-g.accuracy),color=g.background_type.map({"land":colors[0],"water":colors[1]}))
    ax.set_xticks(np.arange(len(g)),[f"{v.background_bank} · {LABELS[v.true_y]}\n{v.background_type}" for v in g.itertuples()],rotation=15)
    ax.set_ylabel("Error rate (%)"); save(fig,"assigned_group_error.png")

    fig,axes=plt.subplots(1,2,figsize=(11,3.7),sharey=True)
    for ax,bank in zip(axes,"AB"):
        for cls,color in zip(LABELS,colors):
            block=influence[(influence.bank==bank)&(influence.omitted_unit=="scene")&(influence.true_class==cls)]
            ax.plot(np.arange(len(block)),block.margin_D,"o-",color=color,label=cls,markersize=3)
        ax.axhline(0,color="black",lw=.6); ax.set_title(f"Bank {bank}"); ax.set_xlabel("Omitted scene index (Land then Water)"); ax.legend()
    axes[0].set_ylabel("Water minus Land margin"); save(fig,"leave_one_scene_out.png")


def run(output):
    settings=json.loads(CONFIG.read_text(encoding="utf-8"))
    fg,bg,pred,arrays=load(output)
    effects=foreground_effects(fg,bg,arrays)
    effects.to_csv(output/"paired_foreground_effects.csv",index=False)
    intervals=bootstrap(fg,bg,arrays,settings)
    intervals.to_csv(output/"effect_intervals.csv",index=False)
    influence=omissions(fg,bg,arrays)
    influence.to_csv(output/"omission_sensitivity.csv",index=False)
    matches=matched_effects(fg,bg,arrays,pd.read_csv(output/"matched_background_subset.csv"))
    matches.to_csv(output/"low_level_matched_effects.csv",index=False)
    group,scenes,category=grouped_outputs(output,fg,bg,pred,effects,arrays)
    make_plots(output,effects,scenes,group,influence)
    slopes=[]
    for cls in (0,1):
        sub=effects.loc[effects.true_y.eq(cls)].pivot(index="source_image_id",columns="bank",values="D")
        slopes.append(dict(true_class=LABELS[cls],spearman_A_B_D=float(spearmanr(sub.A,sub.B).statistic)))
    point=effects.groupby(["bank","true_y"],as_index=False).agg(n=("D","size"),mean_D=("D","mean"),
        median_D=("D","median"),p25_D=("D",lambda x:x.quantile(.25)),p75_D=("D",lambda x:x.quantile(.75)),
        share_D_positive=("D",lambda x:(x>0).mean()),mean_DeltaP=("DeltaP","mean"),median_DeltaP=("DeltaP","median"),
        p25_DeltaP=("DeltaP",lambda x:x.quantile(.25)),p75_DeltaP=("DeltaP",lambda x:x.quantile(.75)),
        switches=("hard_labels",lambda x:(x==2).sum()),median_p_range=("p_range","median"),
        cross_type_flip_rate=("cross_type_flip_rate","mean"),within_land_flip_rate=("within_land_flip_rate","mean"),
        within_water_flip_rate=("within_water_flip_rate","mean"),always_correct=("error_rate",lambda x:(x==0).sum()),
        always_wrong=("error_rate",lambda x:(x==1).sum()))
    point.to_csv(output/"bank_class_summary.csv",index=False)
    write_json(output/"analysis_summary.json",dict(rows=len(pred),birds=len(fg),scenes=len(bg),species=fg.species_id.nunique(),
               replication=slopes, switched_birds_all=int((effects.query("bank=='ALL'").hard_labels==2).sum()),
               all_scene_wga=float(pred.groupby(["true_y","background_type"]).correct.mean().min()),
               analysis_sha256=sha256(__file__), predictions_sha256=sha256(output/"fully_crossed_predictions.csv")))
    print(point.to_string(index=False),flush=True)
    print(intervals.query("method=='crossed_species_scene' and bank!='ALL' and true_class=='Landbird'").to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    run(parser.parse_args().output)
