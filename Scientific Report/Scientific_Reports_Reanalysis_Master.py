#!/usr/bin/env python3
"""Orchestrator for the current Scientific Reports major-revision reanalysis.

Runs the corrected real-data pipeline, nested FP-Cox validation, synthetic-data
fidelity/utility experiment, and optionally the final scikit-survival confirmation.
All scripts must be kept in the same folder as this file.
"""
import argparse, subprocess, sys
from pathlib import Path

def run(cmd):
    print('\n>>>',' '.join(map(str,cmd)))
    subprocess.run(list(map(str,cmd)),check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,default=Path('uci_echocardiogram_reconstructed_132.csv'))
    ap.add_argument('--outdir',type=Path,default=Path('.'))
    ap.add_argument('--standard-confirmation',action='store_true')
    args=ap.parse_args()
    here=Path(__file__).resolve().parent
    data=args.data.resolve(); out=args.outdir.resolve(); out.mkdir(parents=True,exist_ok=True)

    run([sys.executable,here/'Real_Data_Core_Reanalysis.py','--data',data,'--outdir',out/'real_data_core'])
    run([sys.executable,here/'Nested_FP_Cox_Validation_v1.py'])
    run([sys.executable,here/'Synthetic_Augmentation_Experiment.py','--data',data,'--outdir',out/'synthetic_v1','--bandwidth','0.10','--global-ratio','5'])
    if args.standard_confirmation:
        run([sys.executable,here/'Standard_Package_Confirmation_sksurv.py','--data',data,'--outdir',out/'standard_confirmation'])
    else:
        print('\nStandard-package confirmation not run. Re-run with --standard-confirmation after installing scikit-survival.')

if __name__=='__main__': main()
