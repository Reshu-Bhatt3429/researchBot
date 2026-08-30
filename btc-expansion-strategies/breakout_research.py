#!/usr/bin/env python3
"""Event-driven two-sided breakout and failed-breakout tests after predicted expansion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import research as core

ACTIVATIONS = [5, 15, 30]
TRIGGERS = [5, 10, 15, 20, 30]


def simulate(d, signals, h, activation, trigger_bps, style):
    rows=[]; next_allowed=-1
    for t in signals:
        if t < next_allowed or t+h >= len(d): continue
        anchor=d.close.iloc[t]; up=anchor*np.exp(trigger_bps/1e4); down=anchor*np.exp(-trigger_bps/1e4)
        for j in range(t+1,min(t+activation,t+h-1)+1):
            hit_up=d.high.iloc[j]>=up; hit_down=d.low.iloc[j]<=down
            if hit_up and hit_down: break  # OHLC path ambiguous: skip conservatively.
            if not (hit_up or hit_down): continue
            trigger_side=1 if hit_up else -1
            trigger_price=max(up,d.open.iloc[j]) if hit_up else min(down,d.open.iloc[j])
            if style=="continuation": side=trigger_side; entry=trigger_price
            else:
                # Reversal is knowable only after the trigger bar closes back through the anchor.
                rejected=(hit_up and d.close.iloc[j]<anchor) or (hit_down and d.close.iloc[j]>anchor)
                if not rejected: break
                side=-trigger_side; entry=d.close.iloc[j]
            gross=side*np.log(d.close.iloc[t+h]/entry)*1e4
            rows.append({"signal_index":t,"entry_index":j,"exit_index":t+h,"side":side,"gross_bps":gross})
            next_allowed=t+h
            break
    return pd.DataFrame(rows)


def summarize(d,trades,cost,split):
    if trades.empty:
        return {"split":split,"cost_bps":cost,"trade_count":0,"trades_per_day":0,"mean_net_bps":np.nan,"median_net_bps":np.nan,"win_rate":np.nan,"profit_factor":np.nan,"daily_sharpe":np.nan,"max_drawdown":np.nan,"mean_daily_return":np.nan,"standard_error":np.nan,"t_stat":np.nan,"p_value":np.nan,"p_value_two_sided":np.nan,"effective_sample_size":0,"cost_break_even_bps":np.nan}
    net=trades.gross_bps.to_numpy()-cost; dates=d.timestamp.iloc[trades.exit_index].dt.floor("D")
    cal=pd.date_range(d.timestamp.iloc[trades.signal_index.min()].floor("D"),d.timestamp.iloc[trades.exit_index.max()].floor("D"),freq="D",tz="UTC")
    daily=pd.Series(net/1e4,index=dates).groupby(level=0).sum().reindex(cal,fill_value=0)
    se,t,p1,p2=core.hac_inference(daily.to_numpy()); eq=(1+daily).cumprod(); dd=eq/eq.cummax()-1
    pos,neg=net[net>0].sum(),-net[net<0].sum()
    return {"split":split,"cost_bps":cost,"trade_count":len(net),"trades_per_day":len(net)/len(cal),"mean_net_bps":net.mean(),"median_net_bps":np.median(net),"win_rate":np.mean(net>0),"profit_factor":pos/neg if neg else np.nan,"daily_sharpe":daily.mean()/daily.std()*np.sqrt(365) if daily.std()>0 else np.nan,"max_drawdown":dd.min(),"mean_daily_return":daily.mean(),"standard_error":se,"t_stat":t,"p_value":p1,"p_value_two_sided":p2,"effective_sample_size":len(daily),"cost_break_even_bps":trades.gross_bps.mean()}


def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output",required=True);a=p.parse_args()
    d=core.load(Path(a.input));x=core.features(d);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    rows=[]; trade_store={}
    for h in core.HORIZONS:
        y=core.targets(d,h);valid=x.notna().all(axis=1)&y.notna().all(axis=1);masks,aidx,bidx=core.split_masks(valid)
        idxs={k:np.flatnonzero(v) for k,v in masks.items()};idxs["development"]=idxs["development"][idxs["development"]+h<aidx];idxs["confirmation"]=idxs["confirmation"][idxs["confirmation"]+h<bidx]
        model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,l2_regularization=5,learning_rate=.06,random_state=core.SEED)
        model.fit(x.iloc[idxs["development"]],np.log1p(y.abs_return_bps.iloc[idxs["development"]]));pred=np.expm1(model.predict(x))
        for cov in core.COVERAGES:
            threshold=np.quantile(pred[idxs["development"]],1-cov)
            for activation in ACTIVATIONS:
                for trigger in TRIGGERS:
                    for style in ["continuation","reversal"]:
                        family=f"{style}_a{activation}_k{trigger}"
                        for block,idx in idxs.items():
                            sig=idx[pred[idx]>=threshold];tr=simulate(d,sig,h,activation,trigger,style)
                            trade_store[(family,h,cov,block)]=tr
                            for cost in core.COSTS: rows.append({"family":family,"horizon":h,"coverage":cov,"activation":activation,"trigger_bps":trigger,"threshold_bps":threshold,**summarize(d,tr,cost,block)})
    surf=pd.DataFrame(rows);surf.to_csv(out/"breakout_surface.csv",index=False)
    ranked=[]
    for fam in surf.family.unique():
        best=core.neighborhood_score(surf,fam)
        if best: ranked.append({"family":fam,"neighborhood_median_bps":best[0],"neighbor_positive_rate":best[1],"horizon":best[2],"coverage":best[3],"confirmation_bps":best[4]})
    rank=pd.DataFrame(ranked).sort_values(["neighborhood_median_bps","neighbor_positive_rate"],ascending=False);rank.to_csv(out/"breakout_ranking.csv",index=False)
    chosen=[];alltr=[]
    for _,r in rank.head(5).iterrows():
        for block in ["development","confirmation","final"]:
            tr=trade_store[(r.family,int(r.horizon),float(r.coverage),block)]
            for cost in core.COSTS: chosen.append({"family":r.family,"horizon":r.horizon,"coverage":r.coverage,**summarize(d,tr,cost,block)})
            if block=="final" and not tr.empty:
                z=tr.copy();z["strategy"]=r.family;z["net_bps_12"]=z.gross_bps-12;alltr.append(z)
    pd.DataFrame(chosen).to_csv(out/"breakout_selected_metrics.csv",index=False)
    pd.concat(alltr,ignore_index=True).to_csv(out/"breakout_final_trades.csv",index=False) if alltr else None
    print(rank.head(15).to_string(index=False));print("\nFINAL\n",pd.DataFrame(chosen).query("split=='final' and cost_bps==12").sort_values("mean_net_bps",ascending=False).to_string(index=False))


if __name__=="__main__":main()
