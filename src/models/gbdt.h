//
// Created by wuyuncheng on 12/1/20.
//

#ifndef PIVOT_GBDT_H
#define PIVOT_GBDT_H

#include "cart_tree.h"

class GBDT {
public:
    int num_trees;                                          // number of trees for gbdt
    int gbdt_type;                                          // type = 0, classification; type = 1, regression
    int classes_num;                                        // 1 for regression, others for classification
    int forest_size;                                        // size of the forest
    std::vector<float> learning_rates;                      // learning_rates of each tree, defaut 1.0
    //std::vector< std::vector<DecisionTree> > forests;     // for regression, only one forest is enough; for classification, require classes_num forests
    std::vector<DecisionTree> forest;                       // for regression, forest size = num_trees; for classification, require forest_size = num_trees * classes_num
    std::vector< std::vector<float> > training_data;        // training dataset
    std::vector< std::vector<float> > testing_data;         // training dataset
    std::vector<float> training_data_labels;                // labels of training dataset
    std::vector<float> testing_data_labels;                 // labels of testing dataset

public:
    /**
     * default constructor
     */
    GBDT();

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
    GBDT(int m_tree_num,
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
    ~GBDT();

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
     * init each tree during the training stage, using the predicted labels of previous trees
     *
     * @param client
     * @param class_id
     * @param tree_id
     * @param cur_predicted_labels
     */
    void init_single_tree_data(Client & client, int class_id, int tree_id, std::vector<float> cur_predicted_labels);

    /**
     * init each tree during the training stage
     *
     * @param client
     * @param class_id
     * @param tree_id
     */
    void init_simplified_single_tree_data(Client &client, int class_id, int tree_id);

    /**
     * build the gbdt forests
     *
     * @param client
     */
    void build_gbdt(Client & client);

    /**
     * build the gbdt forests with spdz computations
     *
     * @param client
     */
    void build_gbdt_with_spdz(Client & client);

    /**
     * compute squared encrypted label vector via spdz
     *
     * @param client
     * @param squared_label_vector
     * @param encrypted_label_vector
     */
    void compute_squared_label_vector(Client & client, EncodedNumber * & squared_label_vector, EncodedNumber * encrypted_label_vector);

    /**
     * compute softmax encrypted label vector for each class via spdz
     *
     * @param client
     * @param softmax_label_vector
     * @param encrypted_classes_label_vector
     */
    void compute_softmax_label_vector(Client & client, EncodedNumber * & softmax_label_vector, EncodedNumber * encrypted_classes_label_vector);

    /**
     * init root node with two encrypted label vectors
     *
     * @param client
     * @param real_tree_id
     * @param encrypted_label_vector
     * @param encrypted_square_label_vector
     */
    void init_root_node_gbdt(Client & client, int real_tree_id,
            EncodedNumber * encrypted_label_vector, EncodedNumber * encrypted_square_label_vector);

    /**
     * compute encrypted predicted labels
     *
     * @param client
     * @param class_id
     * @param tree_id
     * @param encrypted_predicted_labels
     * @param flag
     */
    void compute_encrypted_predicted_labels(Client & client, int class_id, int tree_id, EncodedNumber *& encrypted_predicted_labels, int flag);

    /**
     * test the accuracy
     *
     * @param client
     * @param accuracy
     */
    void test_accuracy(Client & client, float & accuracy);

    /**
     * compute the data labels for a dataset
     *
     * @param client
     * @param class_id
     * @param tree_id
     * @param flag: 0, training set; 1, testing set
     * @return
     */
    std::vector<float> compute_predicted_labels(Client & client, int class_id, int tree_id, int flag);

    /**
     * predict a result given a sample id
     *
     * @param class_id
     * @param tree_id
     * @param sample_values
     * @param node_index_2_leaf_index_map
     */
    std::vector<int> compute_binary_vector(int class_id, int tree_id, std::vector<float> sample_values, std::map<int, int> node_index_2_leaf_index_map);
};

#endif //PIVOT_GBDT_H
