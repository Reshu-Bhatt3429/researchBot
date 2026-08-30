#!/usr/bin/env python3
"""Exploratory target/stop exits for expansion-gated two-sided breakouts."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import research as core

HORIZONS=[30,60]; COVERAGES=[.10,.05,.02]; ACTIVATIONS=[5,15]; TRIGGERS=[5,10]
TARGETS=[20,30,40,50]; STOPS=[10,20,30]

def simulate(d,signals,h,activation,trigger,target,stop):
    rows=[];next_allowed=-1
    for t in signals:
        if t<next_allowed or t+h>=len(d):continue
        anchor=d.close.iloc[t];up=anchor*np.exp(trigger/1e4);down=anchor*np.exp(-trigger/1e4)
        entry_i=side=entry=None
        for j in range(t+1,min(t+activation,t+h-1)+1):
            hu,hd=d.high.iloc[j]>=up,d.low.iloc[j]<=down
            if hu and hd:break
            if hu:entry_i,side,entry=j,1,max(up,d.open.iloc[j]);break
            if hd:entry_i,side,entry=j,-1,min(down,d.open.iloc[j]);break
        if entry_i is None:continue
        tp=entry*np.exp(side*target/1e4);sl=entry*np.exp(-side*stop/1e4);exit_price=d.close.iloc[t+h];reason="timeout"
        for j in range(entry_i+1,t+h+1):
            ht=(d.high.iloc[j]>=tp) if side>0 else (d.low.iloc[j]<=tp)
            hs=(d.low.iloc[j]<=sl) if side>0 else (d.high.iloc[j]>=sl)
            if ht and hs:exit_price=sl;reason="collision_stop";break
            if hs:exit_price=min(sl,d.open.iloc[j]) if side>0 else max(sl,d.open.iloc[j]);reason="stop";break
            if ht:exit_price=max(tp,d.open.iloc[j]) if side>0 else min(tp,d.open.iloc[j]);reason="target";break
        gross=side*np.log(exit_price/entry)*1e4;rows.append({"signal_index":t,"entry_index":entry_i,"exit_index":j if reason!="timeout" else t+h,"side":side,"gross_bps":gross,"reason":reason});next_allowed=t+h
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();d=core.load(Path(a.input));x=core.features(d);out=Path(a.output);rows=[]
    for h in HORIZONS:
        y=core.targets(d,h);valid=x.notna().all(axis=1)&y.notna().all(axis=1);masks,aidx,bidx=core.split_masks(valid);idxs={k:np.flatnonzero(v) for k,v in masks.items()};idxs['development']=idxs['development'][idxs['development']+h<aidx];idxs['confirmation']=idxs['confirmation'][idxs['confirmation']+h<bidx]
        model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,l2_regularization=5,learning_rate=.06,random_state=core.SEED);model.fit(x.iloc[idxs['development']],np.log1p(y.abs_return_bps.iloc[idxs['development']]));pred=np.expm1(model.predict(x))
        for cov in COVERAGES:
            threshold=np.quantile(pred[idxs['development']],1-cov)
            for activation in ACTIVATIONS:
                for trigger in TRIGGERS:
                    for target in TARGETS:
                        for stop in STOPS:
                            family=f"barrier_a{activation}_k{trigger}_t{target}_s{stop}"
                            for block,idx in idxs.items():
                                tr=simulate(d,idx[pred[idx]>=threshold],h,activation,trigger,target,stop)
                                for cost in core.COSTS:rows.append({"family":family,"horizon":h,"coverage":cov,"activation":activation,"trigger_bps":trigger,"target_bps":target,"stop_bps":stop,**__import__('breakout_research').summarize(d,tr,cost,block)})
    s=pd.DataFrame(rows);s.to_csv(out/'barrier_surface.csv',index=False)
    conf=s[(s.split=='confirmation')&(s.cost_bps==12)&(s.trades_per_day>=1.5)].sort_values('mean_net_bps',ascending=False)
    picks=conf.head(10)[['family','horizon','coverage']];selected=s.merge(picks,on=['family','horizon','coverage']);selected.to_csv(out/'barrier_selected_metrics.csv',index=False)
    print('CONFIRMATION\n',conf.head(15).to_string(index=False));print('\nFINAL (exploratory; family added after prior final inspection)\n',selected.query("split=='final' and cost_bps==12").sort_values('mean_net_bps',ascending=False).to_string(index=False))
if __name__=='__main__':main()
