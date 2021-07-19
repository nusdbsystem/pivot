//
// Created by wuyuncheng on 26/11/19.
//

#include "tree_node.h"
extern FILE * logger_out;

TreeNode::TreeNode() {
    is_leaf = -1;
    depth = -1;
    is_self_feature = -1;
    best_client_id = -1;
    best_feature_id = -1;
    best_split_id = -1;
    type = -1;
    sample_size = -1;
    classes_num = -1;
    left_child = -1;
    right_child = -1;
    available_global_feature_num = -1;
}

TreeNode::TreeNode(int m_depth, int m_type, int m_sample_size, int m_classes_num,
        EncodedNumber *m_sample_iv, EncodedNumber *m_encrypted_labels) {
    is_leaf = -1;
    depth = m_depth;
    type = m_type;
    is_self_feature = -1;
    best_client_id = -1;
    best_feature_id = -1;
    best_split_id = -1;
    sample_size = m_sample_size;
    sample_iv = new EncodedNumber[sample_size];
    for (int i = 0; i < sample_size; i++) {
        sample_iv[i] = m_sample_iv[i];
    }
    classes_num = m_classes_num;
    encrypted_labels = new EncodedNumber[classes_num * sample_size];
    for (int i = 0; i < classes_num * sample_size; i++) {
        encrypted_labels[i] = m_encrypted_labels[i];
    }
    left_child = -1;
    right_child = -1;
    available_global_feature_num = -1;
}

TreeNode::TreeNode(const TreeNode &node) {
    is_leaf = node.is_leaf;
    depth = node.depth;
    is_self_feature = node.is_self_feature;
    best_client_id = node.best_client_id;
    best_feature_id = node.best_feature_id;
    best_split_id = node.best_split_id;
    available_feature_ids = node.available_feature_ids;
    available_global_feature_num = node.available_global_feature_num;
    type = node.type;
    sample_size = node.sample_size;
    classes_num = node.classes_num;
    impurity = node.impurity;
    sample_iv = new EncodedNumber[sample_size];
    for (int i = 0; i < sample_size; i++) {
        sample_iv[i] = node.sample_iv[i];
    }
    encrypted_labels = new EncodedNumber[classes_num * sample_size];
    for (int i = 0; i < classes_num; i++) {
        encrypted_labels[i] = node.encrypted_labels[i];
    }
    label = node.label;
    left_child = node.left_child;
    right_child = node.right_child;
}

TreeNode& TreeNode::operator=(TreeNode *node) {
    is_leaf = node->is_leaf;
    depth = node->depth;
    is_self_feature = node->is_self_feature;
    best_client_id = node->best_client_id;
    best_feature_id = node->best_feature_id;
    best_split_id = node->best_split_id;
    available_feature_ids = node->available_feature_ids;
    available_global_feature_num = node->available_global_feature_num;
    type = node->type;
    sample_size = node->sample_size;
    classes_num = node->classes_num;
    sample_iv = new EncodedNumber[sample_size];
    impurity = node->impurity;
    for (int i = 0; i < sample_size; i++) {
        sample_iv[i] = node->sample_iv[i];
    }
    encrypted_labels = new EncodedNumber[classes_num * sample_size];
    for (int i = 0; i < classes_num; i++) {
        encrypted_labels[i] = node->encrypted_labels[i];
    }
    label = node->label;
    left_child = node->left_child;
    right_child = node->right_child;
}


void TreeNode::print_node() {
    logger(logger_out, "Node depth = %d\n", depth);
    logger(logger_out, "Is leaf = %d\n", is_leaf);
    logger(logger_out, "Is self feature = %d\n", is_self_feature);
    logger(logger_out, "Best client id = %d\n", best_client_id);
    if (is_self_feature) {
        logger(logger_out, "Best feature id = %d\n", best_feature_id);
        logger(logger_out, "Best split id = %d\n", best_split_id);
    }
    if (is_leaf) {
        float f_label;
        label.decode(f_label);
        logger(logger_out, "Label = %f\n", f_label);
    }

}

TreeNode::~TreeNode() {
    if (is_leaf != -1) {
//        bellow two vectors are freed during training the trees to save memory
//        delete [] sample_iv;
//        delete [] encrypted_labels;
    }
}