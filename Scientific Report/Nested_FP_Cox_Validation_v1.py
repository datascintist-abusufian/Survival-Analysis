#!/usr/bin/env python3
from pathlib import Path
import argparse
import json, math
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from statsmodels.duration.hazard_regression import PHReg

SEED=42
PREDICTORS=['age','pericardial_effusion','fractional_shortening','epss','lvdd','wall_motion_index']
FP_POWERS=[-2,-1,-0.5,0,0.5,1,2,3]

def fp(x,p):
    x=np.asarray(x,float)
    if p==0: return np.log(x)
    return x**p

def cindex(time,event,risk):
    time=np.asarray(time,float); event=np.asarray(event,int); risk=np.asarray(risk,float)
    c=0.; n=0.
    for i in range(len(time)):
        for j in range(i+1,len(time)):
            if time[i]==time[j]: continue
            if time[i]<time[j] and event[i]==1:
                n+=1; c += 1 if risk[i]>risk[j] else 0.5 if risk[i]==risk[j] else 0
            elif time[j]<time[i] and event[j]==1:
                n+=1; c += 1 if risk[j]>risk[i] else 0.5 if risk[j]==risk[i] else 0
    return c/n if n else np.nan

def fit_matrix(d,X):
    m=PHReg(d['survival'].astype(float),X.astype(float),status=d['event'].astype(int),ties='breslow')
    r=m.fit(disp=0)
    aic=2*X.shape[1]-2*float(r.llf)
    return r,aic

def design(d,pa,pw):
    X=d[PREDICTORS].copy().astype(float)
    X['age']=fp(d['age'],pa)
    X['wall_motion_index']=fp(d['wall_motion_index'],pw)
    return X

def breslow(train,X,beta):
    T=train.survival.to_numpy(float); E=train.event.to_numpy(int)
    exp_lp=np.exp(X.to_numpy(float)@beta)
    rows=[]; cum=0.
    for t in np.sort(np.unique(T[E==1])):
        dd=np.sum((T==t)&(E==1)); den=np.sum(exp_lp[T>=t])
        if den>0:
            cum += dd/den; rows.append((t,cum))
    return np.asarray(rows,float)

def h0_at(b,t):
    if b.size==0:return 0.
    k=np.searchsorted(b[:,0],t,side='right')-1
    return 0. if k<0 else float(b[k,1])

def pred_surv(train,Xtr,fit,Xte,times):
    beta=np.asarray(fit.params); b=breslow(train,Xtr,beta)
    rr=np.exp(Xte.to_numpy(float)@beta)
    return np.column_stack([np.exp(-h0_at(b,t)*rr) for t in times])

def km_cens(train):
    T=train.survival.to_numpy(float); C=1-train.event.to_numpy(int)
    g=1.; rows=[]
    for t in np.sort(np.unique(T[C==1])):
        nr=np.sum(T>=t); d=np.sum((T==t)&(C==1))
        if nr>0:g*=1-d/nr
        rows.append((t,g))
    return np.asarray(rows,float)

def Gat(km,t,left=False):
    if km.size==0:return 1.
    k=np.searchsorted(km[:,0],t,side='left' if left else 'right')-1
    return 1. if k<0 else max(float(km[k,1]),1e-8)

def brier(train,test,sp,times):
    km=km_cens(train); T=test.survival.to_numpy(float); E=test.event.to_numpy(int); out=[]
    for j,t in enumerate(times):
        terms=[]
        for i in range(len(test)):
            if T[i]<=t and E[i]==1: terms.append((sp[i,j]**2)/Gat(km,T[i],left=True))
            elif T[i]>t: terms.append(((1-sp[i,j])**2)/Gat(km,t))
            else: terms.append(0.)
        out.append(np.mean(terms))
    return np.asarray(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=Path('uci_echocardiogram_reconstructed_132.csv'))
    ap.add_argument('--outdir', type=Path, default=Path('.'))
    args=ap.parse_args()
    base=args.outdir.resolve()
    base.mkdir(parents=True, exist_ok=True)
    df=pd.read_csv(args.data.resolve())
    real=df.dropna(subset=['survival','event']+PREDICTORS).copy().reset_index(drop=True)
    rkf=RepeatedKFold(n_splits=5,n_repeats=5,random_state=SEED)
    horizons=np.asarray([12.,24.,36.]); grid=np.linspace(6.,36.,16)
    rows=[]
    for fold,(tr,te) in enumerate(rkf.split(real)):
        train=real.iloc[tr].reset_index(drop=True); test=real.iloc[te].reset_index(drop=True)
        best=None
        for pa in FP_POWERS:
            for pw in FP_POWERS:
                try:
                    Xtr=design(train,pa,pw); fit,aic=fit_matrix(train,Xtr)
                    if best is None or aic<best['aic']:
                        best={'pa':pa,'pw':pw,'aic':aic,'fit':fit,'Xtr':Xtr}
                except Exception: pass
        if best is None: continue
        Xte=design(test,best['pa'],best['pw'])
        risk=Xte.to_numpy(float)@np.asarray(best['fit'].params)
        c=cindex(test.survival,test.event,risk)
        sp=pred_surv(train,best['Xtr'],best['fit'],Xte,horizons); bs=brier(train,test,sp,horizons)
        spg=pred_surv(train,best['Xtr'],best['fit'],Xte,grid); bsg=brier(train,test,spg,grid)
        ibs=float(np.trapezoid(bsg,grid)/(grid[-1]-grid[0]))
        rows.append({'fold':fold,'age_power':best['pa'],'wmi_power':best['pw'],'training_AIC':best['aic'],
                     'c_index':c,'brier_12m':bs[0],'brier_24m':bs[1],'brier_36m':bs[2],'ibs_6_36m':ibs})
    out=pd.DataFrame(rows); out.to_csv(base/'fp_cox_nested_repeated5fold_5repeats.csv',index=False)
    summ=out[['c_index','brier_12m','brier_24m','brier_36m','ibs_6_36m']].agg(['mean','std']).T
    summ.to_csv(base/'fp_cox_nested_validation_summary.csv')
    powers=out.groupby(['age_power','wmi_power']).size().reset_index(name='fold_count').sort_values('fold_count',ascending=False)
    powers.to_csv(base/'fp_cox_selected_powers_by_fold_summary.csv',index=False)
    print('SUMMARY\n',summ)
    print('\nPOWER SELECTION\n',powers.to_string(index=False))

if __name__=='__main__':main()
