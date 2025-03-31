#!/bin/bash

myArray=("TotalCapture" "EKUT" "ACCAD" "CMU" "KIT" "BMLmovi" "BMLrub" "EyesJapanDataset" "HDM05" "HumanEva" "MoSh" "SSM" "TCDHands" "Transitions" "WEIZMANN")

for dataset in ${myArray[@]}; do
    echo $dataset 
    python ./scripts/preprocess_data.py --dataset ${dataset}
done