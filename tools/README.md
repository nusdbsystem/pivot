## Python dependencies

+ install all dependencies by `pip install -r requirements.txt`

## Major Functionalities

+ DT_RF_GDBT.py - codes to train:
   + classification
      + DecisionTree
      + RandomForest
      + GradientBoosting
   + regression
      + DecisionTree
      + RandomForest
      + GradientBoosting
+ data_generator.py - generate dataset for:
   + classification
   + regression
   

1. install dependencies (sklearn, numpy, argparse)
   + install Anaconda (already include these packages)
      + download:   `wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh`
      + install:    `bash path-to-downloaded.sh file`
2. python DT_RF_GBDT.py --params (run the non-private tree models)
3. python data_generator.py --params (run the synthetic data generators)

