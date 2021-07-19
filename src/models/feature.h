//
// Created by wuyuncheng on 26/11/19.
//

#ifndef PIVOT_FEATURE_H
#define PIVOT_FEATURE_H

#include "../utils/util.h"
#include <iostream>
#include <vector>

class Feature {
public:
    int id;                                           // index of local feature
    int num_splits;                                   // the number of splits of the current feature, should <= max_bins - 1
    int max_bins;                                     // the maximum number of bins
    int is_used;                                      // 0: not used, 1: used, -1: not decided
    int is_categorical;                               // 0: not categorical, 1: categorical, -1: not decided
    std::vector<float> split_values;                  // int or float
    std::vector<float> original_feature_values;       // the original data of this feature in the training dataset
    float maximum_value;                              // maximum value of this feature
    float minimum_value;                              // minimum value of this feature
    std::vector<int> sorted_indexes;                  // the values of this feature are sorted, the re-sorted indexes are recorded
    //std::vector<float> sorted_distinct_values;      // the sorted distinct values after sorting the feature values
    std::vector< std::vector<int> > split_ivs_left;   // pre-compute the 0 1 ivs for each split in the left branch, before re-sorting
    std::vector< std::vector<int> > split_ivs_right;  // pre-compute the 0 1 ivs for each split in the right branch, before re-sorting

public:
    Feature();
    Feature(int m_id, int m_categorical, int m_num_splits, int m_max_bins);
    Feature(int m_id, int m_categorical, int m_num_splits, int m_max_bins, std::vector<float> m_values, int m_size);
    ~Feature();

    /**
     * copy constructor
     *
     * @param feature
     */
    Feature(const Feature & feature);

    /**
     * copy assignment constructor
     *
     * @param feature
     * @return
     */
    Feature &operator = (Feature *feature);

    /**
     * pre-compute and store the split indicator vectors
     * once a split is found, directly using the split iv to update the encrypted sample iv
     */
    void compute_split_ivs();

    /**
     * find the split values given the data and max_bins
     * currently assume the categorical values could also be sorted,
     * i.e., using label encoder instead of one hot encoder
     */
    void find_splits();

    /**
     * sort the feature values to accelerate the computation when computing the encrypted statistics
     * store the sorted indexes in the sorted_indexes vector
     */
    void sort_feature();

    /**
     * helper function, sort indexes given a vector v
     *
     * @param v
     */
    std::vector<int> sort_indexes(const std::vector<float> &v);

    /**
     * set the feature data given a column in the training dataset
     *
     * @param values
     * @param size
     */
    void set_feature_data(std::vector<float> values, int size);


    /**
     * compute distinct values to find if it is categorical or continuous
     * according to the distinct values, update split_num
     *
     * @return
     */
    std::vector<float> compute_distinct_values();

    void test_split_correctness();
};


#endif //PIVOT_FEATURE_H
