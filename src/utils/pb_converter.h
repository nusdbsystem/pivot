//
// Created by wuyuncheng on 18/10/19.
//

#ifndef PIVOT_PB_CONVERTER_H
#define PIVOT_PB_CONVERTER_H

#include <vector>
#include "encoder.h"

/**
 * pb serialize batch ids
 *
 * @param batch_ids
 * @param size
 * @param output_str
 */
void serialize_batch_ids(int *batch_ids, int size, std::string & output_str);

/**
 * pb deserialize batch ids
 *
 * @param batch_ids
 * @param input_str
 */
void deserialize_ids_from_string(int *& batch_ids, std::string input_str);

/**
 * pb serialize encoded number
 *
 * @param number
 * @param output_str
 */
void serialize_encoded_number(EncodedNumber number, std::string & output_str);

/**
 * pb deserialize encoded number
 *
 * @param number
 * @param input_str
 */
void deserialize_number_from_string(EncodedNumber & number, std::string input_str);

/**
 * pb serialize batch sums
 *
 * @param batch_sums
 * @param size
 * @param output_str
 */
void serialize_batch_sums(EncodedNumber *batch_sums, int size, std::string & output_str);

/**
 * pb deserialize batch sums
 *
 * @param batch_sums
 * @param size
 * @param input_str
 */
void deserialize_sums_from_string(EncodedNumber *& batch_sums, int & size, std::string input_str);

/**
 * pb serialize batch losses
 *
 * @param batch_losses
 * @param size
 * @param output_str
 */
void serialize_batch_losses(EncodedNumber *batch_losses, int size, std::string & output_str);

/**
 * pb deserialize batch losses
 *
 * @param batch_losses
 * @param input_str
 */
void deserialize_losses_from_string(EncodedNumber *& batch_losses, std::string input_str);

/**
 * pb serialize pruning condition result
 *
 * @param node_index
 * @param is_satisfied
 * @param encrypted_labels
 * @param classes_num
 * @param sample_num
 * @param label
 * @param output_str
 */
void serialize_pruning_condition_result(int node_index, int is_satisfied, EncodedNumber** encrypted_labels,
        int classes_num, int sample_num, EncodedNumber label, std::string & output_str);

/**
 * pb deserialize pruning condition result
 *
 * @param node_index
 * @param is_satisfied
 * @param encrypted_label_vecs
 * @param label
 * @param input_str
 */
void deserialize_pruning_condition_result(int & node_index, int & is_satisfied,
        EncodedNumber **& encrypted_label_vecs, EncodedNumber & label, std::string input_str);

/**
 * pb serialize encrypted statistics
 *
 * @param client_id
 * @param node_index
 * @param split_num
 * @param classes_num
 * @param left_sample_nums
 * @param right_sample_nums
 * @param encrypted_statistics
 * @param output_str
 */
void serialize_encrypted_statistics(int client_id, int node_index, int split_num, int classes_num,
        EncodedNumber* left_sample_nums, EncodedNumber* right_sample_nums,
        EncodedNumber ** encrypted_statistics, std::string & output_str);

/**
 * pb deserialize encrypted statistics
 *
 * @param client_id
 * @param node_index
 * @param split_num
 * @param classes_num
 * @param left_sample_nums
 * @param right_sample_nums
 * @param encrypted_statistics
 * @param input_str
 */
void deserialize_encrypted_statistics(int & client_id, int & node_index, int & split_num, int & classes_num,
        EncodedNumber * & left_sample_nums, EncodedNumber * & right_sample_nums,
        EncodedNumber ** & encrypted_statistics, std::string input_str);

/**
 * serialize update information
 *
 * @param source_client_id
 * @param best_client_id
 * @param best_feature_id
 * @param best_split_id
 * @param left_branch_impurity
 * @param right_branch_impurity
 * @param left_branch_sample_iv
 * @param right_branch_sample_iv
 * @param output_str
 */
void serialize_update_info(int source_client_id, int best_client_id, int best_feature_id, int best_split_id,
        EncodedNumber left_branch_impurity, EncodedNumber right_branch_impurity,
        EncodedNumber* left_branch_sample_iv, EncodedNumber *right_branch_sample_iv, int sample_size, std::string & output_str);

/**
 * deserialize update information
 *
 * @param source_client_id
 * @param best_client_id
 * @param best_feature_id
 * @param best_split_id
 * @param left_branch_impurity
 * @param right_branch_impurity
 * @param left_branch_sample_iv
 * @param right_branch_sample_iv
 * @param input_str
 */
void deserialize_update_info(int & source_client_id, int & best_client_id, int & best_feature_id, int & best_split_id,
        EncodedNumber & left_branch_impurity, EncodedNumber & right_branch_impurity,
        EncodedNumber* & left_branch_sample_iv, EncodedNumber* & right_branch_sample_iv, std::string input_str);

/**
 * serialize split nums
 *
 * @param global_split_num
 * @param client_split_nums
 * @param output_str
 */
void serialize_split_info(int global_split_num, std::vector<int> client_split_nums, std::string & output_str);

/**
 * deserialize split nums
 *
 * @param global_split_num
 * @param client_split_nums
 * @param input_str
 */
void deserialize_split_info(int & global_split_num, std::vector<int> & client_split_nums, std::string input_str);

/**
 * serialize prune check result
 *
 * @param node_index
 * @param is_satisfied
 * @param label
 * @param output_str
 */
void serialize_prune_check_result(int node_index, int is_satisfied, EncodedNumber label, std::string & output_str);

/**
 * deserialize prune check result
 *
 * @param node_index
 * @param is_satisfied
 * @param label
 * @param input_str
 */
void deserialize_prune_check_result(int & node_index, int & is_satisfied, EncodedNumber & label, std::string input_str);

/**
 * serialize encrypted vector
 *
 * @param node_index
 * @param classes_num
 * @param sample_num
 * @param encrypted_label_vector
 * @param output_str
 */
void serialize_encrypted_label_vector(int node_index, int classes_num,
        int sample_num, EncodedNumber * encrypted_label_vector, std::string & output_str);

/**
 * deserialize encrypted label vector
 *
 * @param node_index
 * @param encrypted_label_vector
 * @param input_str
 */
void deserialize_encrypted_label_vector(int & node_index, EncodedNumber *& encrypted_label_vector, std::string input_str);


#endif //PIVOT_PB_CONVERTER_H
