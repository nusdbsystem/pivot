//
// Created by wuyuncheng on 29/12/19.
//

#ifndef PIVOT_COMMON_H
#define PIVOT_COMMON_H

/*************************** Party parameters *************************/

#define TOTAL_CLIENT_NUM 3
#define SPDZ_PORT_NUM_DT 18000
#define NUM_SPDZ_PARTIES 3
#define SPDZ_PORT_NUM_DT_ENHANCED 19000
#define SPDZ_PORT_NUM_RF_CLASSIFICATION_PREDICTION 20000
#define ROUNDED_PRECISION 1e-3
#define SPDZ_LG2P 128
#define SUPER_CLIENT_ID 0
enum SPDZComputationID {LeafCheck, LeafLabelComp,
    FindBestSplit, GBDTLabelSquare, GBDTSoftmax};

/********************* Decision tree parameters ***********************/

#define FLOAT_PRECISION 8
#define SPDZ_FIXED_PRECISION 8 // should be careful for regression tree
#define PRECISION_THRESHOLD 1e-6
#define GLOBAL_FEATURE_NUM 35
#define MAX_IMPURITY 2.0
#define MAX_VARIANCE 100000.0
#define MAX_GLOBAL_SPLIT_NUM 6000
#define MAX_DEPTH 2
#define MAX_BINS 32
#define DEFAULT_CLASSES_NUM 3
#define TREE_TYPE 0  // 0: classification tree, 1: regression tree
#define PRUNE_SAMPLE_NUM 5
#define PRUNE_VARIANCE_THRESHOLD 0.0001
#define MAXIMUM_RAND_VALUE 32767
#define CLASS_NUM_FOR_REGRESSION 2
enum SolutionType {Basic, Enhanced};
enum OptimizationType {Non, CombiningSplits, Parallelism, All};
enum TreeType {Classification, Regression};

/********************* Random forest parameters ***********************/

#define NUM_TREES 3
#define RF_SAMPLE_RATE 0.8

/************************** GBDT parameters ***************************/

#define GBDT_LEARNING_RATE 1.0
#define SIMULATE_VALUE1 0.25
#define SIMULATE_VALUE2 1.5

/************************* Program parameters *************************/

#define SPLIT_PERCENTAGE 0.8
#define NUM_OMP_THREADS 4
enum Algorithm{DecisionTreeAlg, RandomForestAlg, GBDTAlg};

// ${Program_Home}/../Pivot-SPDZ/Player-Data/3-128-128/Params-Data
#define DEFAULT_PARAM_DATA_FILE "../third_party/Pivot-SPDZ/Player-Data/3-128-128/Params-Data"

#endif //PIVOT_COMMON_H
