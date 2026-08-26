"""Post-process Project A predictions into paired effects and bootstrap CIs."""
from pathlib import Path
import argparse, numpy as np, pandas as pd

COMPLETE={"AD_noncholera","Malaria","ILI","ALRI_u5","Bloody_diarrhoea","Typhoid"}

def boot_ci(x, seed=20260826, B=2000):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)==0:return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed); means=np.empty(B)
    for i in range(B): means[i]=rng.choice(x,size=len(x),replace=True).mean()
    return (float(x.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pred',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out); d=pd.read_csv(a.pred)
    key=["split","condition","district","origin_t","horizon","method"]
    z=d[d['mode']=='zero_fill'][key+['abs_error']].rename(columns={'abs_error':'ae_zero'})
    r=d[d['mode']=='reporting_aware'][key+['abs_error']].rename(columns={'abs_error':'ae_aware'})
    p=z.merge(r,on=key,how='inner'); p['condition_group']=np.where(p.condition.isin(COMPLETE),'complete_six','rotating_stress')
    p['delta_zero_minus_aware']=p.ae_zero-p.ae_aware
    rows=[]
    for cols,g in p.groupby(['split','condition_group','horizon','method']):
        m,lo,hi=boot_ci(g.delta_zero_minus_aware)
        rows.append(dict(zip(['split','condition_group','horizon','method'],cols),n=len(g),mean_delta=m,ci_low=lo,ci_high=hi,zero_mae=g.ae_zero.mean(),aware_mae=g.ae_aware.mean()))
    pd.DataFrame(rows).to_csv(out/'paired_effects_bootstrap.csv',index=False)
    # Macro average: equal weight to district-condition series, not observations.
    g=p[p.split=='test'].groupby(['condition_group','condition','district','horizon','method']).agg(zero_mae=('ae_zero','mean'),aware_mae=('ae_aware','mean'),delta=('delta_zero_minus_aware','mean'),n=('delta_zero_minus_aware','size')).reset_index()
    macro=g.groupby(['condition_group','horizon','method']).agg(series=('delta','size'),macro_zero_mae=('zero_mae','mean'),macro_aware_mae=('aware_mae','mean'),macro_delta=('delta','mean')).reset_index()
    macro.to_csv(out/'macro_series_metrics_test.csv',index=False)
    p.to_csv(out/'paired_predictions.csv',index=False)
    print(pd.DataFrame(rows).query("split=='test'").to_string(index=False))

if __name__=='__main__':main()
