//
// Created by wuyuncheng on 26/11/19.
//

#include "feature.h"
#include <numeric>      // std::iota
#include <algorithm>    // std::sort
extern FILE * logger_out;

Feature::Feature() {}

Feature::Feature(int m_id, int m_categorical, int m_num_splits, int m_max_bins) {
    id = m_id;
    is_used = 0;
    is_categorical = m_categorical;
    num_splits = m_num_splits;
    max_bins = m_max_bins;
}

Feature::Feature(int m_id, int m_categorical, int m_num_splits,
    int m_max_bins, std::vector<float> m_values, int m_size) {
    id = m_id;
    is_used = 0;
    is_categorical = m_categorical;
    num_splits = m_num_splits;
    max_bins = m_max_bins;
    set_feature_data(m_values, m_size);
    sort_feature();
    find_splits();
    compute_split_ivs();
}

Feature::Feature(const Feature &feature) {
    id = feature.id;
    num_splits = feature.num_splits;
    max_bins = feature.max_bins;
    is_used = feature.is_used;
    is_categorical = feature.is_categorical;
    split_values = feature.split_values;
    original_feature_values = feature.original_feature_values;
    maximum_value = feature.maximum_value;
    minimum_value = feature.minimum_value;
    sorted_indexes = feature.sorted_indexes;
    //sorted_distinct_values = feature.sorted_distinct_values;
    split_ivs_left = feature.split_ivs_left;
    split_ivs_right = feature.split_ivs_right;
}

Feature& Feature::operator=(Feature *feature) {
    id = feature->id;
    num_splits = feature->num_splits;
    max_bins = feature->max_bins;
    is_used = feature->is_used;
    is_categorical = feature->is_categorical;
    split_values = feature->split_values;
    original_feature_values = feature->original_feature_values;
    maximum_value = feature->maximum_value;
    minimum_value = feature->minimum_value;
    sorted_indexes = feature->sorted_indexes;
    //sorted_distinct_values = feature.sorted_distinct_values;
    split_ivs_left = feature->split_ivs_left;
    split_ivs_right = feature->split_ivs_right;
}

Feature::~Feature() {}

void Feature::set_feature_data(std::vector<float> values, int size) {
    float max = -10e8;
    float min = 10e8;
    original_feature_values.reserve(size);
    for (int i = 0; i < size; i++) {
        original_feature_values.push_back(values[i]);
        if (values[i] > max) {
            max = values[i];
        }
        if (values[i] < min) {
            min = values[i];
        }
    }
    maximum_value = max;
    minimum_value = min;
}

std::vector<float> Feature::compute_distinct_values() {
    // now the feature values are sorted, the sorted indexes are stored in sorted_indexes
    int sample_num = original_feature_values.size();
    int distinct_value_num = 0;
    std::vector<float> distinct_values;
    for (int i = 0; i < sample_num; i++) {
        // if no value has been added, directly add a value
        if (distinct_value_num == 0) {
            distinct_values.push_back(original_feature_values[sorted_indexes[i]]);
            distinct_value_num++;
        } else {
            if (distinct_values[distinct_value_num-1] == original_feature_values[sorted_indexes[i]]) {
                continue;
            } else {
                distinct_values.push_back(original_feature_values[sorted_indexes[i]]);
                distinct_value_num++;
            }
        }
    }
    return distinct_values;
}

void Feature::find_splits() {
    /// use quantile sketch method, that computes k split values such that
    /// there are k + 1 bins, and each bin has almost same number samples
    /// basically, after sorting the feature values, we compute the size of
    /// each bin, i.e., n_sample_per_bin = n/(k+1), and samples[0:n_sample_per_bin]
    /// is the first bin, while (value[n_sample_per_bin] + value[n_sample_per_bin+1])/2
    /// is the first split value, etc.
    /// note: currently assume that the feature values is sorted, treat categorical
    /// feature as label encoder sortable values
    int n_samples = original_feature_values.size();
    std::vector<float> distinct_values = compute_distinct_values();
    // if distinct values is larger than max_bins + 1, treat as continuous feature
    // otherwise, treat as categorical feature
    if (distinct_values.size() >= max_bins) {
        // treat as continuous feature, find splits using quantile method (might not accurate when the values are imbalanced)
        int n_sample_per_bin = n_samples / (num_splits + 1);
        for (int i = 0; i < num_splits; i++) {
            float split_value_i = (original_feature_values[sorted_indexes[(i + 1) * n_sample_per_bin]]
                                   + original_feature_values[sorted_indexes[(i + 1) * n_sample_per_bin + 1]])/2;
            split_values.push_back(split_value_i);
        }
    }
    else if (distinct_values.size() > 1) {
        // the split values are same as the distinct values
        num_splits = distinct_values.size() - 1;
        for (int i = 0; i < num_splits; i++) {
            split_values.push_back(distinct_values[i]);
        }
    }
    else {
        // the distinct values is equal to 1, which is suspicious for the input dataset
        logger(logger_out, "This feature has only one distinct value, please check it again\n");
        num_splits = distinct_values.size();
        split_values.push_back(distinct_values[0]);
    }
}

std::vector<int> Feature::sort_indexes(const std::vector<float> &v) {
    // initialize original index locations
    std::vector<int> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    // sort indexes based on comparing values in v
    // sort 100000 running time 30ms, sort 10000 running time 3ms
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}

void Feature::sort_feature() {
    sorted_indexes = sort_indexes(original_feature_values);
}

void Feature::compute_split_ivs() {
    split_ivs_left.reserve(num_splits);
    split_ivs_right.reserve(num_splits);
    for (int i = 0; i < num_splits; i++) {
        // read split value i
        float split_value_i = split_values[i];
        int n_samples = original_feature_values.size();
        std::vector<int> indicator_vec_left;
        indicator_vec_left.reserve(n_samples);
        std::vector<int> indicator_vec_right;
        indicator_vec_right.reserve(n_samples);
        for (int j = 0; j < n_samples; j++) {
            if (original_feature_values[j] <= split_value_i) {
                indicator_vec_left.push_back(1);
                indicator_vec_right.push_back(0);
            } else {
                indicator_vec_left.push_back(0);
                indicator_vec_right.push_back(1);
            }
        }
        split_ivs_left.push_back(indicator_vec_left);
        split_ivs_right.push_back(indicator_vec_right);
    }
}

void Feature::test_split_correctness() {
    for (int i = 0; i < num_splits; i++) {
        int left_sum = 0, right_sum = 0;
        for (int j = 0; j < split_ivs_left[0].size(); j++) {
            if (split_ivs_left[i][j] == 1) left_sum += 1;
            if (split_ivs_right[i][j] == 1) right_sum += 1;
        }
        int total_sum = left_sum + right_sum;
        logger(logger_out, "left_sum = %d, right_sum = %d, total_sum = %d\n",
                left_sum, right_sum, total_sum);
    }
}
