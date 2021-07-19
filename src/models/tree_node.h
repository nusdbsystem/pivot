//
// Created by wuyuncheng on 26/11/19.
//

#ifndef PIVOT_TREE_NODE_H
#define PIVOT_TREE_NODE_H

#include "../utils/encoder.h"
#include "../utils/djcs_t_aux.h"
#include "../utils/util.h"
#include <vector>

class TreeNode {
public:
    int is_leaf;                                // 0: not leaf, 1: leaf node, -1: not decided
    int depth;                                  // the depth of the current tree node, root node is 0, -1: not decided
    int is_self_feature;                        // 0: not self feature, 1: self feature, -1: not decided
    int best_client_id;                         // -1: not decided
    int best_feature_id;                        // -1: not self feature, 0 -- d_i: self feature id
    int best_split_id;                          // split_id, -1: not decided
    std::vector<int> available_feature_ids;     // the available local feature ids
    int available_global_feature_num;           // the number of global features globally
    int type;                                   // 0: classification, 1: regression
    int sample_size;                            // the number of samples on the node
    int classes_num;                            // the number of classes on the node
    EncodedNumber impurity;                     // the impurity of the current tree node, Gini index for classification, variance for regression
    EncodedNumber *sample_iv;                   // the encrypted indicator vector of which samples are available on this node
    EncodedNumber *encrypted_labels;            // the encrypted label information, classification: classes_num * sample_num, regression: 2 * sample_num
    EncodedNumber label;                        // if is_leaf is true, a label is assigned
    int left_child;                             // left branch id of the current node, if it is not a leaf node, -1: not decided
    int right_child;                            // right branch id of the current node, if it is not a leaf node, -1: not decided

public:
    TreeNode();
    TreeNode(int m_depth, int type, int sample_size, int m_classes_num, EncodedNumber *sample_iv, EncodedNumber *m_encrypted_labels);
    ~TreeNode();

    void print_node();

    /**
     * copy constructor
     *
     * @param node
     */
    TreeNode(const TreeNode &node);

    /**
     * assignment constructor
     *
     * @param node
     * @return
     */
    TreeNode &operator=(TreeNode *node);
};

struct PredictionObj {
    int is_leaf;
    int is_self_feature;
    int best_client_id;
    int best_feature_id;
    int best_split_id;
    int mark;
    int index;

    PredictionObj() {
        is_leaf = -1;
        is_self_feature = -1;
        best_client_id = -1;
        best_feature_id = -1;
        best_split_id = -1;
        mark = -1;
        index = -1;
    }

    PredictionObj(int m_is_leaf, int m_is_self_feature, int m_best_client_id,
            int m_best_feature_id, int m_best_split_id, int m_mark, int m_index) {
        is_leaf = m_is_leaf;
        is_self_feature = m_is_self_feature;
        best_client_id = m_best_client_id;
        best_feature_id = m_best_feature_id;
        best_split_id = m_best_split_id;
        mark = m_mark;
        index = m_index;
    }

    ~PredictionObj() {}
};

#endif //PIVOT_TREE_NODE_H
