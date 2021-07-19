//
// Created by wuyuncheng on 18/10/19.
//

#include <google/protobuf/io/coded_stream.h>
#include "pb_converter.h"
#include "util.h"
#include "../include/protobuf/common.pb.h"
#include "../include/protobuf/logistic.pb.h"
#include "../include/protobuf/cart.pb.h"
extern FILE * logger_out;

void serialize_encoded_number(EncodedNumber number, std::string & output_str) {
    com::collaborative::ml::PB_EncodedNumber pb_number;
    char * n_str_c, * value_str_c;
    n_str_c = mpz_get_str(NULL, 10, number.n);
    value_str_c = mpz_get_str(NULL, 10, number.value);
    std::string n_str(n_str_c), value_str(value_str_c);
    pb_number.set_n(n_str);
    pb_number.set_value(value_str);
    pb_number.set_exponent(number.exponent);
    pb_number.set_type(number.type);
    pb_number.SerializeToString(& output_str);

    free(n_str_c);
    free(value_str_c);
}

void deserialize_number_from_string(EncodedNumber & number, std::string input_str) {
    com::collaborative::ml::PB_EncodedNumber deserialized_pb_number;
    if (!deserialized_pb_number.ParseFromString(input_str)) {
        logger(logger_out, "Failed to parse PB_EncodedNumber from string\n");
        return;
    }
    mpz_set_str(number.n, deserialized_pb_number.n().c_str(), 10);
    mpz_set_str(number.value, deserialized_pb_number.value().c_str(), 10);
    number.exponent = deserialized_pb_number.exponent();
    number.type = deserialized_pb_number.type();
}

void serialize_batch_ids(int *batch_ids, int size, std::string & output_str) {
    com::collaborative::ml::PB_BatchIds pb_batch_ids;
    for (int i = 0; i < size; i++) {
        pb_batch_ids.add_batch_id(batch_ids[i]);
    }
    pb_batch_ids.SerializeToString(&output_str);

    pb_batch_ids.Clear();
}

void deserialize_ids_from_string(int *& batch_ids, std::string input_str) {
    com::collaborative::ml::PB_BatchIds deserialized_pb_batch_ids;
    if (!deserialized_pb_batch_ids.ParseFromString(input_str)) {
        logger(logger_out, "Failed to parse PB_BatchIds from string\n");
        return;
    }
    for (int i = 0; i < deserialized_pb_batch_ids.batch_id_size(); i++) {
        batch_ids[i] = deserialized_pb_batch_ids.batch_id(i);
    }
}

void serialize_batch_sums(EncodedNumber *batch_sums, int size, std::string & output_str) {
    com::collaborative::ml::PB_BatchSums pb_batch_sums;
    for (int i = 0; i < size; i++) {
        com::collaborative::ml::PB_EncodedNumber *pb_number = pb_batch_sums.add_batch_sum();
        char * n_str_c, * value_str_c;
        n_str_c = mpz_get_str(NULL, 10, batch_sums[i].n);
        value_str_c = mpz_get_str(NULL, 10, batch_sums[i].value);
        std::string n_str(n_str_c), value_str(value_str_c);
        pb_number->set_n(n_str);
        pb_number->set_value(value_str);
        pb_number->set_exponent(batch_sums[i].exponent);
        pb_number->set_type(batch_sums[i].type);

        free(n_str_c);
        free(value_str_c);
    }
    pb_batch_sums.SerializeToString(&output_str);
}

void deserialize_sums_from_string(EncodedNumber *& partial_sums, int & size, std::string input_str) {
    com::collaborative::ml::PB_BatchSums deserialized_batch_partial_sums;
    google::protobuf::io::CodedInputStream inputStream((unsigned char*)input_str.c_str(), input_str.length());
    inputStream.SetTotalBytesLimit(1024*1024*1024, 1024*1024*1024);
    if (!deserialized_batch_partial_sums.ParseFromCodedStream(&inputStream)) {
        logger(logger_out, "Failed to parse PB_BatchPartialSums from string\n");
        return;
    }
    size = deserialized_batch_partial_sums.batch_sum_size();
    partial_sums = new EncodedNumber[size];
    for (int i = 0; i < deserialized_batch_partial_sums.batch_sum_size(); i++) {
        com::collaborative::ml::PB_EncodedNumber pb_number = deserialized_batch_partial_sums.batch_sum(i);
        mpz_set_str(partial_sums[i].n, pb_number.n().c_str(), 10);
        mpz_set_str(partial_sums[i].value, pb_number.value().c_str(), 10);
        partial_sums[i].exponent = pb_number.exponent();
        partial_sums[i].type = pb_number.type();
    }
}

void serialize_batch_losses(EncodedNumber *batch_losses, int size, std::string & output_str) {
    com::collaborative::ml::PB_BatchLosses pb_batch_losses;
    for (int i = 0; i < size; i++) {
        com::collaborative::ml::PB_EncodedNumber *pb_number = pb_batch_losses.add_batch_loss();
        char * n_str_c, * value_str_c;
        n_str_c = mpz_get_str(NULL, 10, batch_losses[i].n);
        value_str_c = mpz_get_str(NULL, 10, batch_losses[i].value);
        std::string n_str(n_str_c), value_str(value_str_c);
        pb_number->set_n(n_str);
        pb_number->set_value(value_str);
        pb_number->set_exponent(batch_losses[i].exponent);
        pb_number->set_type(batch_losses[i].type);

        free(n_str_c);
        free(value_str_c);
    }
    pb_batch_losses.SerializeToString(&output_str);
}

void deserialize_losses_from_string(EncodedNumber *& batch_losses, std::string input_str) {
    com::collaborative::ml::PB_BatchLosses deserialized_batch_losses;
    google::protobuf::io::CodedInputStream inputStream((unsigned char*)input_str.c_str(), input_str.length());
    inputStream.SetTotalBytesLimit(1024*1024*1024, 1024*1024*1024);
    if (!deserialized_batch_losses.ParseFromCodedStream(&inputStream)) {
        logger(logger_out, "Failed to parse PB_BatchPartialSums from string\n");
        return;
    }
    for (int i = 0; i < deserialized_batch_losses.batch_loss_size(); i++) {
        com::collaborative::ml::PB_EncodedNumber pb_number = deserialized_batch_losses.batch_loss(i);
        mpz_set_str(batch_losses[i].n, pb_number.n().c_str(), 10);
        mpz_set_str(batch_losses[i].value, pb_number.value().c_str(), 10);
        batch_losses[i].exponent = pb_number.exponent();
        batch_losses[i].type = pb_number.type();
    }
}

void serialize_pruning_condition_result(int node_index, int is_satisfied, EncodedNumber** encrypted_labels,
        int classes_num, int sample_num, EncodedNumber label, std::string & output_str) {
    com::collaborative::ml::PB_PruneConditionResult pb_pruning_condition_result;
    pb_pruning_condition_result.set_node_index(node_index);
    pb_pruning_condition_result.set_pruning_satisfied(is_satisfied);
    if (is_satisfied == 0) {
        // not satisfied, serialize encrypted label vectors for finding split
        for (int i = 0; i < classes_num; i++) {
            com::collaborative::ml::PB_EncryptedLabelVec *pb_encrypted_label_vec = pb_pruning_condition_result.add_encrypted_label_vec();
            for (int j = 0; j < sample_num; j++) {
                com::collaborative::ml::PB_EncodedNumber *pb_number = pb_encrypted_label_vec->add_encrypted_label();
                char * n_str_c, * value_str_c;
                n_str_c = mpz_get_str(NULL, 10, encrypted_labels[i][j].n);
                value_str_c = mpz_get_str(NULL, 10, encrypted_labels[i][j].value);
                std::string n_str(n_str_c), value_str(value_str_c);
                pb_number->set_n(n_str);
                pb_number->set_value(value_str);
                pb_number->set_exponent(encrypted_labels[i][j].exponent);
                pb_number->set_type(encrypted_labels[i][j].type);

                free(n_str_c);
                free(value_str_c);
            }
        }
    } else {
        // satisfied, serialize encrypted impurity and plaintext label
        com::collaborative::ml::PB_EncodedNumber *pb_number_label = new com::collaborative::ml::PB_EncodedNumber;
        char * n_str_label_c, * value_str_label_c;
        n_str_label_c = mpz_get_str(NULL, 10, label.n);
        value_str_label_c = mpz_get_str(NULL, 10, label.value);
        std::string n_str_label(n_str_label_c), value_str_label(value_str_label_c);
        pb_number_label->set_n(n_str_label);
        pb_number_label->set_value(value_str_label);
        pb_number_label->set_exponent(label.exponent);
        pb_number_label->set_type(label.type);
        pb_pruning_condition_result.set_allocated_label(pb_number_label);

        free(n_str_label_c);
        free(value_str_label_c);
    }
    pb_pruning_condition_result.SerializeToString(&output_str);
}

void deserialize_pruning_condition_result(int & node_index, int & is_satisfied, EncodedNumber **& encrypted_label_vecs,
                                          EncodedNumber & label, std::string input_str) {
    com::collaborative::ml::PB_PruneConditionResult deserialized_pruning_condition_result;
    google::protobuf::io::CodedInputStream inputStream((unsigned char*)input_str.c_str(), input_str.length());
    inputStream.SetTotalBytesLimit(1024*1024*1024, 1024*1024*1024);
    if (!deserialized_pruning_condition_result.ParseFromCodedStream(&inputStream)) {
        logger(logger_out, "Failed to parse PB_PruneConditionResult from string\n");
        return;
    }
    node_index = deserialized_pruning_condition_result.node_index();
    is_satisfied = deserialized_pruning_condition_result.pruning_satisfied();
    if (is_satisfied == 0) {
        // not satisfied, read encrypted labels
        for (int i = 0; i < deserialized_pruning_condition_result.encrypted_label_vec_size(); i++) {
            com::collaborative::ml::PB_EncryptedLabelVec pb_label_vec = deserialized_pruning_condition_result.encrypted_label_vec(i);
            for (int j = 0; j < pb_label_vec.encrypted_label_size(); j++) {
                com::collaborative::ml::PB_EncodedNumber pb_number = pb_label_vec.encrypted_label(j);
                // TODO: WHY NEED INIT HERE, NOT REFERENCE PARAMETER?
//                mpz_init(encrypted_label_vecs[i][j].n);
//                mpz_init(encrypted_label_vecs[i][j].value);
//                gmp_printf("n = %Zd\n", encrypted_label_vecs[i][j].n);
//                gmp_printf("value = %Zd\n", encrypted_label_vecs[i][j].value);
                mpz_set_str(encrypted_label_vecs[i][j].n, pb_number.n().c_str(), 10);
                mpz_set_str(encrypted_label_vecs[i][j].value, pb_number.value().c_str(), 10);
                encrypted_label_vecs[i][j].exponent = pb_number.exponent();
                encrypted_label_vecs[i][j].type = pb_number.type();
            }
        }
    } else {
        // satisfied, read encrypted impurity and label
        com::collaborative::ml::PB_EncodedNumber pb_number_label = deserialized_pruning_condition_result.label();
        mpz_set_str(label.n, pb_number_label.n().c_str(), 10);
        mpz_set_str(label.value, pb_number_label.value().c_str(), 10);
        label.exponent = pb_number_label.exponent();
        label.type = pb_number_label.type();
    }
    logger(logger_out, "Error after return?\n");
}

void serialize_encrypted_statistics(int client_id, int node_index, int split_num, int classes_num,
                               EncodedNumber* left_sample_nums, EncodedNumber* right_sample_nums,
                               EncodedNumber ** encrypted_statistics, std::string & output_str) {
    com::collaborative::ml::PB_EncryptedStatistics pb_encrypted_statistics;
    pb_encrypted_statistics.set_client_id(client_id);
    pb_encrypted_statistics.set_node_index(node_index);
    pb_encrypted_statistics.set_local_split_num(split_num);
    pb_encrypted_statistics.set_classes_num(classes_num);
    if (split_num != 0) {
        // local feature is not used up
        for (int i = 0; i < split_num; i++) {
            com::collaborative::ml::PB_EncodedNumber *pb_number_left = pb_encrypted_statistics.add_left_sample_nums_of_splits();
            com::collaborative::ml::PB_EncodedNumber *pb_number_right = pb_encrypted_statistics.add_right_sample_nums_of_splits();
            char * n_str_left_c, * value_str_left_c, * n_str_right_c, * value_str_right_c;
            n_str_left_c = mpz_get_str(NULL, 10, left_sample_nums[i].n);
            value_str_left_c = mpz_get_str(NULL, 10, left_sample_nums[i].value);
            std::string n_str_left(n_str_left_c), value_str_left(value_str_left_c);
            pb_number_left->set_n(n_str_left);
            pb_number_left->set_value(value_str_left);
            pb_number_left->set_exponent(left_sample_nums[i].exponent);
            pb_number_left->set_type(left_sample_nums[i].type);
            n_str_right_c = mpz_get_str(NULL, 10, right_sample_nums[i].n);
            value_str_right_c = mpz_get_str(NULL, 10, right_sample_nums[i].value);
            std::string n_str_right(n_str_right_c), value_str_right(value_str_right_c);
            pb_number_right->set_n(n_str_right);
            pb_number_right->set_value(value_str_right);
            pb_number_right->set_exponent(right_sample_nums[i].exponent);
            pb_number_right->set_type(right_sample_nums[i].type);

            com::collaborative::ml::PB_EncryptedStatPerSplit *pb_encrypted_stat_split = pb_encrypted_statistics.add_encrypted_stats_of_splits();
            for (int j = 0; j < classes_num * 2; j++) {
                com::collaborative::ml::PB_EncodedNumber *pb_number = pb_encrypted_stat_split->add_encrypted_stat();
                char * n_str_c, * value_str_c;
                n_str_c = mpz_get_str(NULL, 10, encrypted_statistics[i][j].n);
                value_str_c = mpz_get_str(NULL, 10, encrypted_statistics[i][j].value);
                std::string n_str(n_str_c), value_str(value_str_c);
                pb_number->set_n(n_str);
                pb_number->set_value(value_str);
                pb_number->set_exponent(encrypted_statistics[i][j].exponent);
                pb_number->set_type(encrypted_statistics[i][j].type);

                free(n_str_c);
                free(value_str_c);
            }
            free(n_str_left_c);
            free(n_str_right_c);
            free(value_str_left_c);
            free(value_str_right_c);
        }
    }
    pb_encrypted_statistics.SerializeToString(&output_str);
}

void deserialize_encrypted_statistics(int & client_id, int & node_index, int & split_num, int & classes_num,
                                      EncodedNumber * & left_sample_nums, EncodedNumber * & right_sample_nums,
                                      EncodedNumber ** & encrypted_statistics, std::string input_str) {
    com::collaborative::ml::PB_EncryptedStatistics deserialized_encrypted_statistics;
    google::protobuf::io::CodedInputStream inputStream((unsigned char*)input_str.c_str(), input_str.length());
    inputStream.SetTotalBytesLimit(1024*1024*1024, 1024*1024*1024);
    if (!deserialized_encrypted_statistics.ParseFromCodedStream(&inputStream)) {
        logger(logger_out, "Failed to parse PB_EncryptedStatistics from string\n");
        return;
    }
    client_id = deserialized_encrypted_statistics.client_id();
    node_index = deserialized_encrypted_statistics.node_index();
    split_num = deserialized_encrypted_statistics.local_split_num();
    classes_num = deserialized_encrypted_statistics.classes_num();
    if (split_num != 0) {
        left_sample_nums = new EncodedNumber[split_num];
        right_sample_nums = new EncodedNumber[split_num];
        encrypted_statistics = new EncodedNumber*[split_num];
        for (int i = 0; i < split_num; i++) {
            encrypted_statistics[i] = new EncodedNumber[2 * classes_num];
        }
        // has encrypted statistics
        for (int i = 0; i < deserialized_encrypted_statistics.left_sample_nums_of_splits_size(); i++) {
            com::collaborative::ml::PB_EncodedNumber pb_number = deserialized_encrypted_statistics.left_sample_nums_of_splits(i);
            mpz_set_str(left_sample_nums[i].n, pb_number.n().c_str(), 10);
            mpz_set_str(left_sample_nums[i].value, pb_number.value().c_str(), 10);
            left_sample_nums[i].exponent = pb_number.exponent();
            left_sample_nums[i].type = pb_number.type();
        }
        for (int i = 0; i < deserialized_encrypted_statistics.right_sample_nums_of_splits_size(); i++) {
            com::collaborative::ml::PB_EncodedNumber pb_number = deserialized_encrypted_statistics.right_sample_nums_of_splits(i);
            mpz_set_str(right_sample_nums[i].n, pb_number.n().c_str(), 10);
            mpz_set_str(right_sample_nums[i].value, pb_number.value().c_str(), 10);
            right_sample_nums[i].exponent = pb_number.exponent();
            right_sample_nums[i].type = pb_number.type();
        }
        for (int i = 0; i < deserialized_encrypted_statistics.encrypted_stats_of_splits_size(); i++) {
            com::collaborative::ml::PB_EncryptedStatPerSplit pb_stat_split = deserialized_encrypted_statistics.encrypted_stats_of_splits(i);
            for (int j = 0; j < pb_stat_split.encrypted_stat_size(); j++) {
                com::collaborative::ml::PB_EncodedNumber pb_number = pb_stat_split.encrypted_stat(j);
                mpz_set_str(encrypted_statistics[i][j].n, pb_number.n().c_str(), 10);
                mpz_set_str(encrypted_statistics[i][j].value, pb_number.value().c_str(), 10);
                encrypted_statistics[i][j].exponent = pb_number.exponent();
                encrypted_statistics[i][j].type = pb_number.type();
            }
        }
    }
}

void serialize_update_info(int source_client_id, int best_client_id, int best_feature_id, int best_split_id,
                           EncodedNumber left_branch_impurity, EncodedNumber right_branch_impurity,
                           EncodedNumber* left_branch_sample_iv, EncodedNumber *right_branch_sample_iv,
                           int sample_size, std::string & output_str) {
    com::collaborative::ml::PB_UpdateInfo pb_update_info;
    pb_update_info.set_source_client_id(source_client_id);
    pb_update_info.set_best_client_id(best_client_id);
    pb_update_info.set_best_feature_id(best_feature_id);
    pb_update_info.set_best_split_id(best_split_id);
    com::collaborative::ml::PB_EncodedNumber *pb_left_impurity = new com::collaborative::ml::PB_EncodedNumber;
    com::collaborative::ml::PB_EncodedNumber *pb_right_impurity = new com::collaborative::ml::PB_EncodedNumber;
    char * n_str_left_c, * value_str_left_c, * n_str_right_c, * value_str_right_c;
    n_str_left_c = mpz_get_str(NULL, 10, left_branch_impurity.n);
    value_str_left_c = mpz_get_str(NULL, 10, left_branch_impurity.value);
    std::string n_str_left(n_str_left_c), value_str_left(value_str_left_c);
    pb_left_impurity->set_n(n_str_left);
    pb_left_impurity->set_value(value_str_left);
    pb_left_impurity->set_exponent(left_branch_impurity.exponent);
    pb_left_impurity->set_type(left_branch_impurity.type);
    n_str_right_c = mpz_get_str(NULL, 10, right_branch_impurity.n);
    value_str_right_c = mpz_get_str(NULL, 10, right_branch_impurity.value);
    std::string n_str_right(n_str_right_c), value_str_right(value_str_right_c);
    pb_right_impurity->set_n(n_str_right);
    pb_right_impurity->set_value(value_str_right);
    pb_right_impurity->set_exponent(right_branch_impurity.exponent);
    pb_right_impurity->set_type(right_branch_impurity.type);
    pb_update_info.set_allocated_left_branch_impurity(pb_left_impurity);
    pb_update_info.set_allocated_right_branch_impurity(pb_right_impurity);
    for (int i = 0; i < sample_size; i++) {
        com::collaborative::ml::PB_EncodedNumber *pb_left_sample_iv = pb_update_info.add_left_branch_sample_ivs();
        com::collaborative::ml::PB_EncodedNumber *pb_right_sample_iv = pb_update_info.add_right_branch_sample_ivs();
        char * n_str_left_sample_c, * value_str_left_sample_c, * n_str_right_sample_c, * value_str_right_sample_c;
        n_str_left_sample_c = mpz_get_str(NULL, 10, left_branch_sample_iv[i].n);
        value_str_left_sample_c = mpz_get_str(NULL, 10, left_branch_sample_iv[i].value);
        std::string n_str_left_sample(n_str_left_sample_c), value_str_left_sample(value_str_left_sample_c);
        pb_left_sample_iv->set_n(n_str_left_sample);
        pb_left_sample_iv->set_value(value_str_left_sample);
        pb_left_sample_iv->set_exponent(left_branch_sample_iv[i].exponent);
        pb_left_sample_iv->set_type(left_branch_sample_iv[i].type);
        n_str_right_sample_c = mpz_get_str(NULL, 10, right_branch_sample_iv[i].n);
        value_str_right_sample_c = mpz_get_str(NULL, 10, right_branch_sample_iv[i].value);
        std::string n_str_right_sample(n_str_right_sample_c), value_str_right_sample(value_str_right_sample_c);
        pb_right_sample_iv->set_n(n_str_right_sample);
        pb_right_sample_iv->set_value(value_str_right_sample);
        pb_right_sample_iv->set_exponent(right_branch_sample_iv[i].exponent);
        pb_right_sample_iv->set_type(right_branch_sample_iv[i].type);

        free(n_str_left_sample_c);
        free(value_str_left_sample_c);
        free(n_str_right_sample_c);
        free(value_str_right_sample_c);
    }
    pb_update_info.SerializeToString(&output_str);

    free(n_str_left_c);
    free(value_str_left_c);
    free(n_str_right_c);
    free(value_str_right_c);
}

void deserialize_update_info(int & source_client_id, int & best_client_id, int & best_feature_id, int & best_split_id,
                             EncodedNumber & left_branch_impurity, EncodedNumber & right_branch_impurity,
                             EncodedNumber* & left_branch_sample_iv, EncodedNumber* & right_branch_sample_iv, std::string input_str) {
    com::collaborative::ml::PB_UpdateInfo deserialized_update_info;
    google::protobuf::io::CodedInputStream inputStream((unsigned char*)input_str.c_str(), input_str.length());
    inputStream.SetTotalBytesLimit(1024*1024*1024, 1024*1024*1024);
    if (!deserialized_update_info.ParseFromCodedStream(&inputStream)) {
        logger(logger_out, "Failed to parse PB_UpdateInfo from string\n");
        return;
    }
    source_client_id = deserialized_update_info.source_client_id();
    best_client_id = deserialized_update_info.best_client_id();
    best_feature_id = deserialized_update_info.best_feature_id();
    best_split_id = deserialized_update_info.best_split_id();
    com::collaborative::ml::PB_EncodedNumber pb_number_left = deserialized_update_info.left_branch_impurity();
    com::collaborative::ml::PB_EncodedNumber pb_number_right = deserialized_update_info.right_branch_impurity();
    mpz_set_str(left_branch_impurity.n, pb_number_left.n().c_str(), 10);
    mpz_set_str(left_branch_impurity.value, pb_number_left.value().c_str(), 10);
    left_branch_impurity.exponent = pb_number_left.exponent();
    left_branch_impurity.type = pb_number_left.type();
    mpz_set_str(right_branch_impurity.n, pb_number_right.n().c_str(), 10);
    mpz_set_str(right_branch_impurity.value, pb_number_right.value().c_str(), 10);
    right_branch_impurity.exponent = pb_number_right.exponent();
    right_branch_impurity.type = pb_number_right.type();
    int sample_size = deserialized_update_info.left_branch_sample_ivs_size();
    left_branch_sample_iv = new EncodedNumber[sample_size];
    right_branch_sample_iv = new EncodedNumber[sample_size];
    for (int i = 0; i < sample_size; i++) {
        com::collaborative::ml::PB_EncodedNumber pb_number_left_iv = deserialized_update_info.left_branch_sample_ivs(i);
        com::collaborative::ml::PB_EncodedNumber pb_number_right_iv = deserialized_update_info.right_branch_sample_ivs(i);
        mpz_set_str(left_branch_sample_iv[i].n, pb_number_left_iv.n().c_str(), 10);
        mpz_set_str(left_branch_sample_iv[i].value, pb_number_left_iv.value().c_str(), 10);
        left_branch_sample_iv[i].exponent = pb_number_left_iv.exponent();
        left_branch_sample_iv[i].type = pb_number_left_iv.type();
        mpz_set_str(right_branch_sample_iv[i].n, pb_number_right_iv.n().c_str(), 10);
        mpz_set_str(right_branch_sample_iv[i].value, pb_number_right_iv.value().c_str(), 10);
        right_branch_sample_iv[i].exponent = pb_number_right_iv.exponent();
        right_branch_sample_iv[i].type = pb_number_right_iv.type();
    }
}

void serialize_split_info(int global_split_num, std::vector<int> client_split_nums, std::string & output_str) {
    com::collaborative::ml::PB_SplitInfo pb_split_info;
    pb_split_info.set_global_split_num(global_split_num);
    for (int i = 0; i < client_split_nums.size(); i++) {
        pb_split_info.add_split_num_vec(client_split_nums[i]);
    }
    pb_split_info.SerializeToString(&output_str);
}

void deserialize_split_info(int & global_split_num, std::vector<int> & client_split_nums, std::string input_str) {
    com::collaborative::ml::PB_SplitInfo deserialized_split_info;
    google::protobuf::io::CodedInputStream inputStream((unsigned char*)input_str.c_str(), input_str.length());
    inputStream.SetTotalBytesLimit(1024*1024*1024, 1024*1024*1024);
    if (!deserialized_split_info.ParseFromCodedStream(&inputStream)) {
        logger(logger_out, "Failed to parse PB_SplitInfo from string\n");
        return;
    }
    global_split_num = deserialized_split_info.global_split_num();
    for (int i = 0; i < deserialized_split_info.split_num_vec_size(); i++) {
        client_split_nums.push_back(deserialized_split_info.split_num_vec(i));
    }
}

void serialize_prune_check_result(int node_index, int is_satisfied, EncodedNumber label, std::string & output_str) {
    com::collaborative::ml::PB_PruneCheckResult pb_prune_check_result;
    pb_prune_check_result.set_node_index(node_index);
    pb_prune_check_result.set_is_satisfied(is_satisfied);
    com::collaborative::ml::PB_EncodedNumber *pb_number_label = new com::collaborative::ml::PB_EncodedNumber;
    char * n_str_label_c, * value_str_label_c;
    n_str_label_c = mpz_get_str(NULL, 10, label.n);
    value_str_label_c = mpz_get_str(NULL, 10, label.value);
    std::string n_str_label(n_str_label_c), value_str_label(value_str_label_c);
    pb_number_label->set_n(n_str_label);
    pb_number_label->set_value(value_str_label);
    pb_number_label->set_exponent(label.exponent);
    pb_number_label->set_type(label.type);
    pb_prune_check_result.set_allocated_label(pb_number_label);
    pb_prune_check_result.SerializeToString(&output_str);

    free(n_str_label_c);
    free(value_str_label_c);
}

void deserialize_prune_check_result(int & node_index, int & is_satisfied, EncodedNumber & label, std::string input_str) {
    com::collaborative::ml::PB_PruneCheckResult deserialize_prune_check_result;
    google::protobuf::io::CodedInputStream inputStream((unsigned char*)input_str.c_str(), input_str.length());
    inputStream.SetTotalBytesLimit(1024*1024*1024, 1024*1024*1024);
    if (!deserialize_prune_check_result.ParseFromCodedStream(&inputStream)) {
        logger(logger_out, "Failed to parse PB_PruneCheckResult from string\n");
        return;
    }
    node_index = deserialize_prune_check_result.node_index();
    is_satisfied = deserialize_prune_check_result.is_satisfied();
    com::collaborative::ml::PB_EncodedNumber pb_number_label = deserialize_prune_check_result.label();
    mpz_set_str(label.n, pb_number_label.n().c_str(), 10);
    mpz_set_str(label.value, pb_number_label.value().c_str(), 10);
    label.exponent = pb_number_label.exponent();
    label.type = pb_number_label.type();
}

void serialize_encrypted_label_vector(int node_index, int classes_num,
                                      int sample_num, EncodedNumber * encrypted_label_vector, std::string & output_str) {
    com::collaborative::ml::PB_EncryptedLabelVector pb_encrypted_label_vector;
    for (int i = 0; i < classes_num * sample_num; i++) {
        com::collaborative::ml::PB_EncodedNumber *pb_number = pb_encrypted_label_vector.add_label_indicator();
        char * n_str_c, * value_str_c;
        n_str_c = mpz_get_str(NULL, 10, encrypted_label_vector[i].n);
        value_str_c = mpz_get_str(NULL, 10, encrypted_label_vector[i].value);
        std::string n_str(n_str_c), value_str(value_str_c);
        pb_number->set_n(n_str);
        pb_number->set_value(value_str);
        pb_number->set_exponent(encrypted_label_vector[i].exponent);
        pb_number->set_type(encrypted_label_vector[i].type);

        free(n_str_c);
        free(value_str_c);
    }
    pb_encrypted_label_vector.SerializeToString(&output_str);
}

void deserialize_encrypted_label_vector(int & node_index, EncodedNumber *& encrypted_label_vector, std::string input_str) {
    com::collaborative::ml::PB_EncryptedLabelVector pb_deserialize_encrypted_label_vector;
    google::protobuf::io::CodedInputStream inputStream((unsigned char*)input_str.c_str(), input_str.length());
    inputStream.SetTotalBytesLimit(1024*1024*1024, 1024*1024*1024);
    if (!pb_deserialize_encrypted_label_vector.ParseFromCodedStream(&inputStream)) {
        logger(logger_out, "Failed to parse PB_EncryptedLabelVector from string\n");
        return;
    }
    encrypted_label_vector = new EncodedNumber[pb_deserialize_encrypted_label_vector.label_indicator_size()];
    for (int i = 0; i < pb_deserialize_encrypted_label_vector.label_indicator_size(); i++) {
        com::collaborative::ml::PB_EncodedNumber pb_number = pb_deserialize_encrypted_label_vector.label_indicator(i);
        mpz_set_str(encrypted_label_vector[i].n, pb_number.n().c_str(), 10);
        mpz_set_str(encrypted_label_vector[i].value, pb_number.value().c_str(), 10);
        encrypted_label_vector[i].exponent = pb_number.exponent();
        encrypted_label_vector[i].type = pb_number.type();
    }
}