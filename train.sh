#! /bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
textpath=SHELL_FOLDER"/text"
files=$(ls $textpath)
for filename in files
do
	/bin/bash -c ". ~/anaconda3/bin/activate py27; python ./camr/amr_parsing.py -m preprocess ./text/$(filename) python ./camr/amr_parsing.py -m parse --model ./camr/model/amr-anno-1.0/amr-anno-1.0.train.basic-abt-brown-verb.m ./text/$(filename); . ~/anaconda3/bin/deactivate"
	python "$(SHELL_FOLDER)/subgraph/amr2subgraph.py" "$(textpath)/$(filename)"
	python "$(SHELL_FOLDER)/feature_extract/tuple.py" "$(SHELL_FOLDER)/subgraph/$(filename).sg"
	python "$(SHELL_FOLDER)/feature_extract/feature.py" "$(SHELL_FOLDER)/feature_extract/$(filename)tuple.pkl"
	# python "$(SHELL_FOLDER)/pred.py" -m "$(SHELL_FOLDER)/classifier.pkl" "$(SHELL_FOLDER)/feature_extract/$(filename)data.pkl"
done

	



