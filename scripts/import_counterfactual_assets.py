"""Explicit import routes for local Places backgrounds or existing prediction CSVs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.counterfactual.common import DEFAULT_DATA
from src.counterfactual.data import import_backgrounds
from src.counterfactual.importers import import_prediction_csv

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=['backgrounds','predictions'])
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    p.add_argument('--mapping',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.kind=='backgrounds':
        result=import_backgrounds(a.data_root,a.manifest)
    else:
        if not a.mapping or not a.output:
            p.error('predictions requires --mapping and --output')
        result=import_prediction_csv(a.manifest,a.mapping,a.data_root/'metadata.csv',a.output)
    print(json.dumps(result,indent=2))
