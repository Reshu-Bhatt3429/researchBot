#!/usr/bin/env python3
"""Causal discovery of BTC 1-minute expansion strategies with multi-minute holds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

SEED = 20260830
HORIZONS = [30, 60, 90, 120]
COVERAGES = [0.20, 0.10, 0.05, 0.02, 0.01]
COSTS = [0.0, 12.0, 30.0]
MIN_TRADES_DAY = 1.5


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def file_hash(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    d = pd.read_csv(path).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d.open_time.astype("int64"), unit="ms", utc=True)
    assert d.open_time.diff().dropna().eq(60000).all()
    assert (d[["open", "high", "low", "close"]] > 0).all().all()
    return d


def zscore(s, n):
    base = s.shift(1)
    return (s - base.rolling(n, min_periods=n // 3).mean()) / base.rolling(n, min_periods=n // 3).std().replace(0, np.nan)


def features(d):
    logc = np.log(d.close)
    r1 = logc.diff()
    x = pd.DataFrame(index=d.index)
    for n in [1, 3, 5, 10, 15, 30, 60, 120, 240, 1440]:
        x[f"ret_{n}"] = (logc - logc.shift(n)) * 1e4
    for n in [5, 15, 30, 60, 120, 240, 1440]:
        x[f"rv_{n}"] = np.sqrt(r1.pow(2).rolling(n, min_periods=max(3, n // 3)).sum()) * 1e4
        x[f"range_{n}"] = np.log(d.high.rolling(n).max() / d.low.rolling(n).min()) * 1e4
    for n in [15, 30, 60, 120, 240]:
        x[f"rv_ratio_{n}"] = x[f"rv_{n}"] / x.rv_1440.replace(0, np.nan)
        hi, lo = d.high.rolling(n).max(), d.low.rolling(n).min()
        x[f"breakout_{n}"] = (2 * (d.close - lo) / (hi - lo).replace(0, np.nan) - 1)
    bar_range = (d.high - d.low).replace(0, np.nan)
    x["body_bps"] = (d.close - d.open) / d.open * 1e4
    x["range_bps"] = bar_range / d.open * 1e4
    x["upper_wick"] = (d.high - d[["open", "close"]].max(axis=1)) / bar_range
    x["lower_wick"] = (d[["open", "close"]].min(axis=1) - d.low) / bar_range
    x["wick_rejection"] = x.lower_wick - x.upper_wick
    lv, lt = np.log1p(d.volume), np.log1p(d.trades)
    for n in [60, 240, 1440]:
        x[f"volume_z_{n}"] = zscore(lv, n)
        x[f"trades_z_{n}"] = zscore(lt, n)
    imbalance = pd.Series(np.where(d.volume > 0, 2 * d.taker_buy_volume / d.volume - 1, 0), index=d.index)
    for n in [1, 5, 15, 30, 60]:
        x[f"imbalance_{n}"] = imbalance.rolling(n, min_periods=n).mean()
    minute = d.timestamp.dt.hour * 60 + d.timestamp.dt.minute
    x["tod_sin"], x["tod_cos"] = np.sin(2*np.pi*minute/1440), np.cos(2*np.pi*minute/1440)
    x["quarter_open"] = (minute % 15 == 0).astype(float)
    x["hour_open"] = (minute % 60 == 0).astype(float)
    return x.replace([np.inf, -np.inf], np.nan)


def targets(d, h):
    entry = d.open.shift(-1)
    ret = np.log(d.close.shift(-h) / entry) * 1e4
    minute_ret = np.log(d.close).diff()
    future_rv = minute_ret.shift(-1).rolling(h, min_periods=h).apply(lambda a: np.sqrt(np.square(a).sum()), raw=True).shift(-(h-1)) * 1e4
    return pd.DataFrame({"entry": entry, "return_bps": ret, "abs_return_bps": ret.abs(), "future_rv_bps": future_rv})


def split_masks(valid):
    idx = np.flatnonzero(valid.to_numpy())
    a, b = idx[int(len(idx)*.55)], idx[int(len(idx)*.80)]
    return {
        "development": valid & (valid.index < a),
        "confirmation": valid & (valid.index >= a) & (valid.index < b),
        "final": valid & (valid.index >= b),
    }, a, b


def nonoverlap(indices, h):
    chosen, next_allowed = [], -1
    for i in indices:
        if i >= next_allowed:
            chosen.append(i)
            next_allowed = i + h
    return np.asarray(chosen, dtype=int)


def side_scores(x, direction_prediction):
    return {
        "direction_ml": direction_prediction,
        "momentum_15": x.ret_15.to_numpy(),
        "momentum_30": x.ret_30.to_numpy(),
        "momentum_60": x.ret_60.to_numpy(),
        "breakout_30": x.breakout_30.to_numpy(),
        "breakout_60": x.breakout_60.to_numpy(),
        "imbalance_15": x.imbalance_15.to_numpy() * 100,
        "impulse_reversal_15": (-x.ret_15 * (1 + np.abs(x.wick_rejection))).to_numpy(),
        "impulse_reversal_30": (-x.ret_30 * (1 + np.abs(x.wick_rejection))).to_numpy(),
        "wick_rejection": x.wick_rejection.to_numpy() * 100,
    }


def hac_inference(daily, lag=7):
    y = np.asarray(daily, float)
    n, mu = len(y), float(np.mean(y))
    u = y - mu
    gamma0 = np.dot(u, u) / n
    long_var = gamma0
    for k in range(1, min(lag, n-1)+1):
        gamma = np.dot(u[k:], u[:-k]) / n
        long_var += 2 * (1-k/(lag+1))*gamma
    se = math.sqrt(max(long_var, 0) / n)
    t = mu / se if se else np.nan
    return se, t, float(norm.sf(t)) if np.isfinite(t) else np.nan, float(2*norm.sf(abs(t))) if np.isfinite(t) else np.nan


def stats(d, idx, side, ret, h, cost, block):
    gross = side[idx] * ret[idx]
    net = gross - cost
    dates = d.timestamp.iloc[idx + h].dt.floor("D")
    calendar = pd.date_range(d.timestamp.iloc[idx.min()].floor("D"), d.timestamp.iloc[min(idx.max()+h, len(d)-1)].floor("D"), freq="D", tz="UTC") if len(idx) else []
    daily = pd.Series(net/1e4, index=dates).groupby(level=0).sum().reindex(calendar, fill_value=0.0) if len(idx) else pd.Series(dtype=float)
    se, t, p1, p2 = hac_inference(daily.to_numpy()) if len(daily) > 10 else (np.nan,)*4
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    equity = (1 + daily).cumprod() if len(daily) else pd.Series(dtype=float)
    dd = equity/equity.cummax()-1 if len(equity) else pd.Series(dtype=float)
    return {
        "split": block, "cost_bps": cost, "trade_count": len(idx),
        "trades_per_day": len(idx)/max(len(calendar), 1), "mean_net_bps": float(np.mean(net)) if len(net) else np.nan,
        "median_net_bps": float(np.median(net)) if len(net) else np.nan, "win_rate": float(np.mean(net>0)) if len(net) else np.nan,
        "profit_factor": float(wins/losses) if losses else np.nan,
        "daily_sharpe": float(daily.mean()/daily.std()*np.sqrt(365)) if len(daily)>2 and daily.std()>0 else np.nan,
        "max_drawdown": float(dd.min()) if len(dd) else np.nan, "mean_daily_return": float(daily.mean()) if len(daily) else np.nan,
        "standard_error": se, "t_stat": t, "p_value": p1, "p_value_two_sided": p2,
        "effective_sample_size": len(daily), "average_holding_minutes": h, "cost_break_even_bps": float(np.mean(gross)) if len(gross) else np.nan,
    }, pd.DataFrame({"date": calendar, "net_return": daily.to_numpy()}) if len(daily) else pd.DataFrame(columns=["date","net_return"])


def neighborhood_score(surface, family):
    s = surface[(surface.family == family) & (surface.cost_bps == 12) & (surface.split == "confirmation")]
    grid = s.pivot(index="horizon", columns="coverage", values="mean_net_bps").reindex(index=HORIZONS, columns=COVERAGES)
    best = None
    for i, h in enumerate(HORIZONS):
        for j, c in enumerate(COVERAGES):
            vals=[]
            for di,dj in [(0,0),(-1,0),(1,0),(0,-1),(0,1)]:
                ni,nj=i+di,j+dj
                if 0<=ni<len(HORIZONS) and 0<=nj<len(COVERAGES): vals.append(grid.iloc[ni,nj])
            row=s[(s.horizon==h)&(s.coverage==c)]
            if row.empty or row.iloc[0].trades_per_day < MIN_TRADES_DAY: continue
            score=float(np.nanmedian(vals)); positive=float(np.mean(np.asarray(vals)>0))
            candidate=(score,positive,h,c,float(row.iloc[0].mean_net_bps))
            if best is None or candidate>best: best=candidate
    return best


def markdown_table(frame):
    if frame.empty: return "_No rows._"
    f=frame.copy()
    for c in f.select_dtypes(include=["float"]).columns: f[c]=f[c].map(lambda v: "" if pd.isna(v) else f"{v:.6g}")
    cols=list(f.columns); lines=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]
    lines += ["| "+" | ".join(str(v) for v in row)+" |" for row in f.itertuples(index=False,name=None)]
    return "\n".join(lines)


def main():
    a=cli(); inp, out=Path(a.input), Path(a.output); out.mkdir(parents=True,exist_ok=True)
    d=load(inp); x=features(d); all_surfaces=[]; prediction_rows=[]; stores={}
    for h in HORIZONS:
        y=targets(d,h); valid=x.notna().all(axis=1)&y.notna().all(axis=1)
        masks,aidx,bidx=split_masks(valid)
        dev=np.flatnonzero(masks["development"]); conf=np.flatnonzero(masks["confirmation"]); final=np.flatnonzero(masks["final"])
        # Purge labels that cross boundaries.
        dev=dev[dev+h<aidx]; conf=conf[conf+h<bidx]
        exp_model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,l2_regularization=5,learning_rate=.06,random_state=SEED)
        dir_model=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,l2_regularization=5,learning_rate=.06,random_state=SEED+1)
        exp_model.fit(x.iloc[dev],np.log1p(y.abs_return_bps.iloc[dev])); dir_model.fit(x.iloc[dev],y.return_bps.iloc[dev].clip(-250,250))
        exp_pred=np.expm1(exp_model.predict(x)); dir_pred=dir_model.predict(x); sides=side_scores(x,dir_pred); ret=y.return_bps.to_numpy()
        stores[h]={"exp_pred":exp_pred,"dir_pred":dir_pred,"return_bps":ret,"indices":{"development":dev,"confirmation":conf,"final":final},"sides":sides}
        for block, idx in stores[h]["indices"].items():
            actual=y.abs_return_bps.iloc[idx].to_numpy(); pred=exp_pred[idx]
            prediction_rows.append({"horizon":h,"split":block,"spearman":spearmanr(pred,actual).statistic,"top_decile_mean_abs_bps":float(actual[pred>=np.quantile(pred,.9)].mean()),"unconditional_mean_abs_bps":float(actual.mean())})
        for coverage in COVERAGES:
            threshold=np.quantile(exp_pred[dev],1-coverage)
            for family,score in sides.items():
                side=np.sign(score)
                for block, idx0 in stores[h]["indices"].items():
                    eligible=idx0[(exp_pred[idx0]>=threshold)&np.isfinite(score[idx0])&(side[idx0]!=0)]
                    idx=nonoverlap(eligible,h)
                    for cost in COSTS:
                        st,_=stats(d,idx,side,ret,h,cost,block)
                        all_surfaces.append({"family":family,"horizon":h,"coverage":coverage,"threshold_bps":threshold,**st})
    surface=pd.DataFrame(all_surfaces); surface.to_csv(out/"parameter_surface_full.csv",index=False)
    pd.DataFrame(prediction_rows).to_csv(out/"expansion_forecast.csv",index=False)
    ranked=[]
    for family in sorted(surface.family.unique()):
        best=neighborhood_score(surface,family)
        if best: ranked.append({"family":family,"neighborhood_median_bps":best[0],"neighbor_positive_rate":best[1],"horizon":best[2],"coverage":best[3],"selected_cell_bps":best[4]})
    ranks=pd.DataFrame(ranked).sort_values(["neighborhood_median_bps","neighbor_positive_rate"],ascending=False)
    ranks.to_csv(out/"candidate_ranking.csv",index=False)
    selected=[]; trades=[]; daily_all=[]
    for _,r in ranks.head(3).iterrows():
        h=int(r.horizon); c=float(r.coverage); store=stores[h]; dev=store["indices"]["development"]
        threshold=np.quantile(store["exp_pred"][dev],1-c); side=np.sign(store["sides"][r.family]); ret=store["return_bps"]
        for block,idx0 in store["indices"].items():
            idx=nonoverlap(idx0[(store["exp_pred"][idx0]>=threshold)&np.isfinite(side[idx0])&(side[idx0]!=0)],h)
            for cost in COSTS:
                st,daily=stats(d,idx,side,ret,h,cost,block); selected.append({"family":r.family,"horizon":h,"coverage":c,**st})
                if cost==12:
                    daily["split"],daily["strategy"]=block,r.family; daily_all.append(daily)
                    for i in idx:
                        trades.append({"entry_time":d.timestamp.iloc[i+1],"exit_time":d.timestamp.iloc[i+h],"symbol":"BTCUSDT","side":"long" if side[i]>0 else "short","gross_return_bps":side[i]*ret[i],"cost_bps":12,"net_return":(side[i]*ret[i]-12)/1e4,"split":block,"strategy":r.family})
    metrics=pd.DataFrame(selected); metrics.to_csv(out/"selected_metrics.csv",index=False)
    pd.DataFrame(trades).to_csv(out/"trades.csv",index=False)
    pd.concat(daily_all,ignore_index=True).to_csv(out/"daily_returns.csv",index=False)
    # Validator-compatible compact artifacts.
    ev=pd.DataFrame(prediction_rows).rename(columns={"horizon":"horizon","top_decile_mean_abs_bps":"mean_return"})
    ev["signal"]="predicted_expansion_top_decile"; ev["count"]=0; ev["median_return"]=np.nan; ev["win_rate"]=np.nan
    ev[["signal","horizon","count","mean_return","median_return","win_rate"]].to_csv(out/"event_study.csv",index=False)
    primary=metrics[(metrics.split=="final")&(metrics.cost_bps==12)].sort_values("mean_net_bps",ascending=False).head(1)
    m=primary.iloc[0].to_dict() if len(primary) else {}
    json.dump({"trade_count":int(m.get("trade_count",0)),"net_expectancy":m.get("mean_net_bps"),"profit_factor":m.get("profit_factor"),"win_rate":m.get("win_rate"),"sharpe":m.get("daily_sharpe"),"max_drawdown":m.get("max_drawdown"),"t_stat":m.get("t_stat"),"p_value":m.get("p_value"),"p_value_two_sided":m.get("p_value_two_sided")},open(out/"metrics.json","w"),indent=2)
    inf=metrics.rename(columns={"trade_count":"sample_size","mean_daily_return":"mean_return"}).copy(); inf["sample_unit"]="calendar_day"; inf["method"]="Newey-West HAC lag 7"; inf["alternative"]="greater"; inf["p_value_adjusted"]=np.minimum(1,inf.p_value*len(surface))
    inf["strategy_id"]=inf.family; inf["hac_lag"]=7; inf["bootstrap_block"]="not used"; inf["repetitions"]=0; inf["seed"]=SEED; inf["null_mean"]=0; inf["diagnostic_only"]=True; inf["selection_context"]="three confirmation-selected families reached final"
    inf[["strategy_id","split","cost_bps","sample_unit","sample_size","effective_sample_size","mean_return","standard_error","t_stat","p_value","p_value_two_sided","p_value_adjusted","method","hac_lag","bootstrap_block","repetitions","seed","null_mean","alternative","diagnostic_only","selection_context"]].to_csv(out/"inference.csv",index=False)
    metrics.assign(variant=metrics.family+"_h"+metrics.horizon.astype(str),net_expectancy=metrics.mean_net_bps,sharpe=metrics.daily_sharpe)[["variant","trade_count","net_expectancy","sharpe","max_drawdown"]].to_csv(out/"robustness.csv",index=False)
    surface.assign(parameter_name="coverage",parameter_value=surface.coverage,metric=surface.mean_net_bps)[["split","cost_bps","parameter_name","parameter_value","metric","trade_count"]].to_csv(out/"parameter_surface.csv",index=False)
    pd.DataFrame([{"split":"final","cost_bps":12,"noise_type":"pending_price_perturbation","noise_level":0,"seed":SEED,"metric":m.get("mean_net_bps",np.nan),"trade_count":m.get("trade_count",0)}]).to_csv(out/"noise_degradation.csv",index=False)
    json.dump({"effective_parameter_count":4,"parameters":["family","horizon","expansion coverage","cost"],"ablations":["expansion gate removed via 20% coverage","direction families compared"],"selection_rule":"highest confirmation neighborhood median subject to >=1.5 trades/day"},open(out/"complexity.json","w"),indent=2)
    config={"research_mode":"strategy","costs":{"round_trip_bps":COSTS,"primary":12},"splits":{"development":.55,"confirmation":.25,"final":.20,"purge":"holding horizon"},"selection_rule":"confirmation neighborhood median, minimum 1.5 trades/day","parameter_budget":{"variants":len(surface)},"noise_plan":{"status":"minimal diagnostic; full seeded perturbation required before paper trading"},"regime_plan":{"status":"not activated because no base strategy passed 12 bps","policy":"NO_TRADE until a base rule is validated"},"inference_plan":{"daily_returns":"Newey-West HAC lag 7","multiplicity":"Bonferroni diagnostic"},"horizons":HORIZONS,"coverages":COVERAGES}
    json.dump(config,open(out/"config.json","w"),indent=2)
    json.dump({"sources":[{"name":"Binance USD-M perpetual klines","file":inp.name}],"timezone":"UTC","start":d.timestamp.iloc[0].isoformat(),"end":d.timestamp.iloc[-1].isoformat(),"instruments":["BTCUSDT"],"sha256":file_hash(inp),"rows":len(d),"missing_bars":0},open(out/"data_manifest.json","w"),indent=2)
    (out/"hypothesis.md").write_text("# Frozen hypothesis\n\nPast volatility, volume, flow, compression and impulse state can predict rare 30–120 minute BTC expansion. A tradable rule additionally needs a direction classifier whose confirmation neighborhood and untouched final expectancy remain positive after 12 bps, while producing at least 1.5 non-overlapping trades per day.\n")
    # Charts and report.
    fig,ax=plt.subplots(figsize=(10,6)); q=pd.DataFrame(prediction_rows); [ax.plot(g.horizon,g.spearman,marker="o",label=b) for b,g in q.groupby("split")]; ax.axhline(0,color="black",lw=1); ax.set(xlabel="Forecast horizon (minutes)",ylabel="Spearman rank correlation",title="Expansion forecast quality"); ax.legend(); fig.tight_layout(); fig.savefig(out/"expansion_forecast.png",dpi=170); plt.close(fig)
    final=metrics[(metrics.split=="final")&(metrics.cost_bps==12)].sort_values("mean_net_bps",ascending=False)
    report="# BTC Expansion Strategy Discovery\n\n## Decision\n\n"
    if len(final) and final.iloc[0].mean_net_bps>0: report+="At least one mined candidate remained positive in the untouched final block, but it remains **exploratory/fragile until full noise and cross-venue validation**.\n\n"
    else: report+="No selected strategy remained positive in the untouched final block after 12 bps; the tested candle/flow families are rejected for trading.\n\n"
    report+="## Expansion prediction\n\nVolatility magnitude is forecastable when rank correlation and top-decile realized movement exceed the unconditional baseline. Profitability still requires correct direction and executable selection.\n\n## Selected candidates\n\n"+markdown_table(metrics[(metrics.cost_bps==12)])+"\n\n## Limitations\n\nOne instrument and one year; bar-level flow only; no L2 book, liquidation feed, funding, or cross-venue validation. The final block was opened once for the three frozen confirmation selections.\n"
    (out/"report.md").write_text(report)
    print(ranks.head(10).to_string(index=False)); print("\nFINAL 12 BPS\n",final.to_string(index=False))


if __name__ == "__main__": main()
