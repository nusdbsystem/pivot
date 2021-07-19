#ifndef PIVOT_RANDOM_FOREST_H
#define PIVOT_RANDOM_FOREST_H

#include "cart_tree.h"

class RandomForest {
public:
    int num_trees;
    std::vector<DecisionTree> forest;
    std::vector< std::vector<float> > training_data;   // training dataset
    std::vector< std::vector<float> > testing_data;    // training dataset
    std::vector<float> training_data_labels;           // labels of training dataset
    std::vector<float> testing_data_labels;           // labels of testing dataset

public:
    /**
     * default constructor
     */
    RandomForest();

    /**
     * constructor
     *
     * @param m_tree_num
     * @param m_global_feature_num
     * @param m_local_feature_num
     * @param m_internal_node_num
     * @param m_type
     * @param m_classes_num
     * @param m_max_depth
     * @param m_max_bins
     * @param m_prune_sample_num
     * @param m_prune_threshold
     * @param solution_type
     * @param optimization_type
     */
    RandomForest(int m_tree_num,
            int m_global_feature_num,
            int m_local_feature_num,
            int m_internal_node_num,
            int m_type,
            int m_classes_num,
            int m_max_depth,
            int m_max_bins,
            int m_prune_sample_num,
            float m_prune_threshold,
            int solution_type,
            int optimization_type);

    /**
     * destructor
     */
    ~RandomForest();

    /**
     * init training data and test data according to split fraction
     * call by client 0
     *
     * @param client
     * @param split
     */
    void init_datasets(Client & client, float split);

    /**
     * init training data and test data according to new indexes received
     *
     * @param client
     * @param new_indexes
     * @param split
     */
    void init_datasets_with_indexes(Client & client, int new_indexes[], float split);

    /**
     * shuffle and assign training data to a decision tree of random forest
     * 
     * @param tree_id
     * @param client
     * @param sample_rate
     */
    void shuffle_and_assign_training_data(int tree_id, Client & client, float sample_rate);

    /**
     * shuffle and assign training data to a decision tree of random forest according to new indexes received
     *
     * @param tree_id
     * @param client
     * @param new_indexes
     * @param sample_rate
     */
    void shuffle_and_assign_training_data_with_indexes(int tree_id, Client & client, int new_indexes[], float sample_rate);

    /**
     * build each tree of random forest
     * 
     * @param client
     * @param sample_rate
     */
    void build_forest(Client & client, float sample_rate);

    /**
     * predict a result given a sample id
     *
     * @param tree_id
     * @param sample_id
     * @param node_index_2_leaf_index_map
     */
    std::vector<int> compute_binary_vector(int tree_id, int sample_id, std::map<int, int> node_index_2_leaf_index_map);

    /**
     * test the accuracy on the test data
     *
     * @param client
     * @param accuracy
     */
    void test_accuracy(Client & client, float & accuracy);

    /**
     * test the accuracy on the test data, using spdz to find the majority class for classification
     *
     * @param client
     * @param accuracy
     */
    void test_accuracy_with_spdz(Client & client, float & accuracy);
};

#endif //PIVOT_RANDOM_FOREST_H
