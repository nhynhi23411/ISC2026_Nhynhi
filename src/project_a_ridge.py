"""Dependency-light pooled lag-ridge benchmark for Project A."""
from pathlib import Path
import argparse, numpy as np, pandas as pd

COMPLETE=["AD_noncholera","Malaria","ILI","ALRI_u5","Bloody_diarrhoea","Typhoid"]
LAGS=[1,2,4,8,12]

def ridge_fit_predict(X,y,x,alpha=10.0):
    X=np.asarray(X,float); y=np.asarray(y,float); x=np.asarray(x,float)
    mu=X.mean(0); sd=X.std(0); sd[sd<1e-8]=1
    Xs=(X-mu)/sd; xs=(x-mu)/sd
    A=Xs.T@Xs + alpha*np.eye(Xs.shape[1]); b=Xs.T@y
    try: beta=np.linalg.solve(A,b)
    except np.linalg.LinAlgError: beta=np.linalg.lstsq(A,b,rcond=None)[0]
    return float(xs@beta)

def build_features(vals,seen,t,mode):
    arr=vals.copy()
    if mode=='zero_fill': arr[~seen|np.isnan(arr)]=0
    else:
        last=0.; out=[]
        for i,v in enumerate(arr):
            if seen[i] and np.isfinite(v): last=float(v)
            out.append(last)
        arr=np.asarray(out)
    f=[]
    for lag in LAGS: f.append(arr[t-lag] if t>=lag else 0.)
    f.append(float(np.mean(arr[max(0,t-4):t])) if t else 0.)
    f.extend([np.sin(2*np.pi*(t%52)/52),np.cos(2*np.pi*(t%52)/52)])
    return f

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--panel',required=True); ap.add_argument('--out',required=True); ap.add_argument('--last-origins',type=int,default=20); a=ap.parse_args()
    out=Path(a.out); d=pd.read_csv(a.panel); d['key']=d.district+'||'+d.condition
    n=int(d.t.max()+1); test_start=n-20; rows=[]
    for condition in d.condition.unique():
        sub=d[d.condition==condition]
        series=[]
        for district,g in sub.groupby('district'):
            g=g.sort_values('t'); vals=np.full(n,np.nan); seen=np.zeros(n,bool)
            vals[g.t.astype(int)]=g.cases.to_numpy(float); seen[g.t.astype(int)]=g.condition_present.to_numpy(bool)&g.row_present.to_numpy(bool)
            series.append((district,vals,seen))
        for mode in ['zero_fill','reporting_aware']:
            for h in [1,2,4]:
                for t in range(max(20,n-a.last_origins),n-h):
                    tt=t+h-1; X=[]; y=[]
                    for district,vals,seen in series:
                        target=vals[:t]; ok=seen[:t]&np.isfinite(target) if mode=='reporting_aware' else np.ones(t,bool)
                        for j in np.where(ok)[0]:
                            if j<12: continue
                            X.append(build_features(vals,seen,j,mode)); y.append(0. if mode=='zero_fill' and (not seen[j] or not np.isfinite(target[j])) else target[j])
                    if len(y)<50: continue
                    for district,vals,seen in series:
                        if not (seen[tt] and np.isfinite(vals[tt])): continue
                        pred=ridge_fit_predict(X,y,build_features(vals,seen,tt,mode))
                        rows.append({'condition':condition,'district':district,'origin_t':t,'horizon':h,'split':'test' if t>=test_start else 'validation','mode':mode,'method':'pooled_ridge','target':vals[tt],'prediction':pred,'abs_error':abs(vals[tt]-pred),'sq_error':(vals[tt]-pred)**2})
    r=pd.DataFrame(rows); r.to_csv(out/'ridge_predictions.csv',index=False)
    m=r.groupby(['split','condition','horizon','mode']).agg(n=('abs_error','size'),MAE=('abs_error','mean'),RMSE=('sq_error',lambda s:float(np.sqrt(s.mean())))).reset_index(); m.to_csv(out/'ridge_metrics.csv',index=False)
    print(m[m.split=='test'].to_string(index=False))
if __name__=='__main__':main()
