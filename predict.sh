#/bin/bash
source ./py27/bin/activate
mode=$1
filename=$2
echo $mode,$filename
python "./subgraph/amr2subgraph.py" $filename
python "./feature_extract/tuple.py" $mode $filename
python "./feature_extract/feature.py" $mode $filename
python "./train/predict.py" $filename
