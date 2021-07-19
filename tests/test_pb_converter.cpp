//
// Created by wuyuncheng on 18/10/19.
//

#include "test_pb_converter.h"
#include <string>
#include <fstream>
#include <sstream>

extern hcs_random *hr;
extern djcs_t_public_key *pk;
extern djcs_t_private_key *vk;
//static djcs_t_auth_server **au = (djcs_t_auth_server **)malloc(TOTAL_CLIENT_NUM * sizeof(djcs_t_auth_server *));
//static mpz_t *si = (mpz_t *)malloc(TOTAL_CLIENT_NUM * sizeof(mpz_t));
extern djcs_t_auth_server **au;
extern mpz_t *si;
extern mpz_t n, positive_threshold, negative_threshold;
extern int total_cases_num, passed_cases_num;
extern FILE * logger_out;

void test_pb_encode_number() {
    EncodedNumber number;
    number.set_float(n, 0.123456);
    number.type = Plaintext;
    std::string s;
    serialize_encoded_number(number, s);
    EncodedNumber deserialized_number;
    deserialize_number_from_string(deserialized_number, s);
    // test equals
    if (number.exponent == deserialized_number.exponent && number.type == deserialized_number.type
            && mpz_cmp(number.n, deserialized_number.n) == 0 && mpz_cmp(number.value, deserialized_number.value) == 0) {
        total_cases_num += 1;
        passed_cases_num += 1;
        logger(logger_out, "test_pb_encode_number: succeed\n");
    } else {
        total_cases_num += 1;
        logger(logger_out, "test_pb_encode_number: failed\n");
    }
}

void test_pb_batch_ids() {
    int *batch_ids = new int[5];
    for (int i = 0; i < 5; i++) {
        batch_ids[i] = i + 1;
    }
    std::string s;
    serialize_batch_ids(batch_ids, 5, s);
    int *deserialized_batch_ids = new int[5];
    deserialize_ids_from_string(deserialized_batch_ids, s);
    // test equals
    bool is_success = true;
    for (int i = 0; i < 5; i++) {
        if (batch_ids[i] != deserialized_batch_ids[i]) {
            is_success = false;
            break;
        }
    }
    if (is_success) {
        total_cases_num += 1;
        passed_cases_num += 1;
        logger(logger_out, "test_pb_batch_ids: succeed\n");
    } else {
        total_cases_num += 1;
        logger(logger_out, "test_pb_batch_ids: failed\n");
    }
}

void test_pb_batch_sums() {
    EncodedNumber *batch_sums = new EncodedNumber[2];
    batch_sums[0].set_float(n, 0.123456);
    batch_sums[0].type = Plaintext;
    batch_sums[1].set_float(n, 0.654321);
    batch_sums[1].type = Ciphertext;
    std::string s;
    serialize_batch_sums(batch_sums, 2, s);
    EncodedNumber *deserialized_partial_sums;// = new EncodedNumber[2];
    int x;
    deserialize_sums_from_string(deserialized_partial_sums, x, s);
    // test equals
    bool is_success = true;
    for (int i = 0; i < 2; i++) {
        if (batch_sums[i].exponent == deserialized_partial_sums[i].exponent
            && batch_sums[i].type == deserialized_partial_sums[i].type
            && mpz_cmp(batch_sums[i].n, deserialized_partial_sums[i].n) == 0
            && mpz_cmp(batch_sums[i].value, deserialized_partial_sums[i].value) == 0) {
            continue;
        } else {
            is_success = false;
        }
    }
    if (is_success) {
        total_cases_num += 1;
        passed_cases_num += 1;
        logger(logger_out, "test_pb_partial_sums: succeed\n");
    } else {
        total_cases_num += 1;
        logger(logger_out, "test_pb_partial_sums: failed\n");
    }
}

void test_pb_batch_losses() {
    EncodedNumber *batch_losses = new EncodedNumber[2];
    batch_losses[0].set_float(n, 0.123456);
    batch_losses[0].type = Plaintext;
    batch_losses[1].set_float(n, 0.654321);
    batch_losses[1].type = Ciphertext;
    std::string s;
    serialize_batch_losses(batch_losses, 2, s);
    EncodedNumber *deserialized_batch_losses = new EncodedNumber[2];
    deserialize_losses_from_string(deserialized_batch_losses, s);
    // test equals
    bool is_success = true;
    for (int i = 0; i < 2; i++) {
        if (batch_losses[i].exponent == deserialized_batch_losses[i].exponent
            && batch_losses[i].type == deserialized_batch_losses[i].type
            && mpz_cmp(batch_losses[i].n, deserialized_batch_losses[i].n) == 0
            && mpz_cmp(batch_losses[i].value, deserialized_batch_losses[i].value) == 0) {
            continue;
        } else {
            is_success = false;
        }
    }
    if (is_success) {
        total_cases_num += 1;
        passed_cases_num += 1;
        logger(logger_out, "test_pb_batch_losses: succeed\n");
    } else {
        total_cases_num += 1;
        logger(logger_out, "test_pb_batch_losses: failed\n");
    }
}

void test_pb_pruning_condition_result() {
    // test is_satisfied false
    int node_index = 0;
    int is_satisfied = 0;
    int classes_num = 2;
    int sample_num = 280;
    EncodedNumber **encrypted_label_vecs = new EncodedNumber*[classes_num];
    for (int i = 0; i < classes_num; i++) {
        encrypted_label_vecs[i] = new EncodedNumber[sample_num];
    }
    for (int i = 0; i < classes_num; i++) {
        for (int j = 0; j < sample_num; j++) {
            encrypted_label_vecs[i][j].set_float(n, 0.555555);
            mpz_set_str(encrypted_label_vecs[i][j].value, "32261786664472188447435471297840865576802914651075700211112206749228131161008036474115527816235662607110055570578029957195401423350983732172758340075092132836458217250595354444037524931458393359651249635626416862339759408107794126657374557936593403426532309083110874705705605329317738804916871097498227225008179399198744183231619738589960224443254991203136470232414614955364113454685818748373592994502880633426768131884099166025439026392901862353585356098058546185861223736070963213524892045587462796813498605146759943290968446712498081499532822515873673486888566969210045479702691653701032085389663765040834761628684", 10);
            encrypted_label_vecs[i][j].type = Ciphertext;
        }
    }

    EncodedNumber label;
    std::string s;
    serialize_pruning_condition_result(node_index, is_satisfied, encrypted_label_vecs, classes_num, sample_num, label, s);
    int d_node_index, d_is_satisfied;
    EncodedNumber **d_encrypted_label_vecs = new EncodedNumber*[classes_num];
    for (int i = 0; i < classes_num; i++) {
        d_encrypted_label_vecs[i] = new EncodedNumber[sample_num];
    }
    EncodedNumber d_label;
    deserialize_pruning_condition_result(d_node_index, d_is_satisfied, d_encrypted_label_vecs, d_label, s);
    // test equals
    bool is_success = true;
    for (int i = 0; i < classes_num; i++) {
        for (int j = 0; j < sample_num; j++) {
            if (d_encrypted_label_vecs[i][j].exponent == encrypted_label_vecs[i][j].exponent
                && d_encrypted_label_vecs[i][j].type == encrypted_label_vecs[i][j].type
                && mpz_cmp(d_encrypted_label_vecs[i][j].n, encrypted_label_vecs[i][j].n) == 0
                && mpz_cmp(d_encrypted_label_vecs[i][j].value, encrypted_label_vecs[i][j].value) == 0) {
                continue;
            } else {
                is_success = false;
            }
        }
    }
    is_success = is_success && d_node_index == node_index && d_is_satisfied == is_satisfied;
    if (is_success) {
        total_cases_num += 1;
        passed_cases_num += 1;
        logger(logger_out, "test_pb_pruning_condition_result: succeed\n");
    } else {
        total_cases_num += 1;
        logger(logger_out, "test_pb_pruning_condition_result: failed\n");
    }
    // test is_satisfied true
    node_index = 1;
    is_satisfied = 1;
    EncodedNumber label_true;
    label_true.set_float(n, 0.555555);
    label_true.type = Plaintext;
    std::string ss;
    serialize_pruning_condition_result(node_index, is_satisfied, encrypted_label_vecs, classes_num, sample_num, label_true, ss);
    deserialize_pruning_condition_result(d_node_index, d_is_satisfied, d_encrypted_label_vecs, d_label, ss);
    // test equals
    is_success = true;
    if (d_label.exponent == label_true.exponent
        && d_label.type == label_true.type
        && mpz_cmp(d_label.n, label_true.n) == 0
        && mpz_cmp(d_label.value, label_true.value) == 0) {
        is_success = is_success;
    } else {
        is_success = is_success && false;
    }
    is_success = is_success && d_node_index == node_index && d_is_satisfied == is_satisfied;
    if (is_success) {
        total_cases_num += 1;
        passed_cases_num += 1;
        logger(logger_out, "test_pb_pruning_condition_result: succeed\n");
    } else {
        total_cases_num += 1;
        logger(logger_out, "test_pb_pruning_condition_result: failed\n");
    }
}

void test_pb_encrypted_statistics() {
    // test split_num = 0
    int client_id = 1;
    int node_index = 2;
    int split_num = 0;
    int classes_num = 0;
    EncodedNumber* left_samples_nums;
    EncodedNumber* right_sample_nums;
    EncodedNumber** encrypted_statistics;
    std::string s, ds;
    serialize_encrypted_statistics(client_id, node_index, split_num, classes_num,
            left_samples_nums, right_sample_nums, encrypted_statistics, s);
    int recv_client_id;
    int recv_node_index;
    int recv_split_num;
    int recv_classes_num;
    deserialize_encrypted_statistics(recv_client_id, recv_node_index, recv_split_num, recv_classes_num,
            left_samples_nums, right_sample_nums, encrypted_statistics, s);
    bool is_success = true;
    if (!(recv_client_id == client_id && recv_node_index == node_index
        && recv_split_num == split_num && recv_classes_num == classes_num)) {
        is_success = false;
    }
    if (is_success) {
        total_cases_num += 1;
        passed_cases_num += 1;
        logger(logger_out, "test_pb_encrypted_statistics: succeed\n");
    } else {
        total_cases_num += 1;
        logger(logger_out, "test_pb_encrypted_statistics: failed\n");
    }

    // test split_num != 0
    split_num = 2;
    classes_num = 1;
    EncodedNumber **encrypted_statistics_x = new EncodedNumber*[split_num];
    for (int i = 0; i < split_num; i++) {
        encrypted_statistics_x[i] = new EncodedNumber[2 * classes_num];
    }
    encrypted_statistics_x[0][0].set_float(n, 0.123456);
    encrypted_statistics_x[0][0].type = Ciphertext;
    encrypted_statistics_x[0][1].set_float(n, 0.222222);
    encrypted_statistics_x[0][1].type = Ciphertext;
    encrypted_statistics_x[1][0].set_float(n, 0.555555);
    encrypted_statistics_x[1][0].type = Ciphertext;
    encrypted_statistics_x[1][1].set_float(n, 0.888888);
    encrypted_statistics_x[1][1].type = Ciphertext;

    EncodedNumber* left_samples_nums_x = new EncodedNumber[split_num];
    EncodedNumber* right_sample_nums_x = new EncodedNumber[split_num];
    left_samples_nums_x[0].set_float(n, 0.123456);
    left_samples_nums_x[0].type = Ciphertext;
    left_samples_nums_x[1].set_float(n, 0.222222);
    left_samples_nums_x[1].type = Ciphertext;
    right_sample_nums_x[0].set_float(n, 0.123456);
    right_sample_nums_x[0].type = Ciphertext;
    right_sample_nums_x[1].set_float(n, 0.222222);
    right_sample_nums_x[1].type = Ciphertext;
    serialize_encrypted_statistics(client_id, node_index, split_num, classes_num,
            left_samples_nums_x, right_sample_nums_x, encrypted_statistics_x, ds);

    EncodedNumber* recv_left_sample_nums;
    EncodedNumber* recv_right_sample_nums;
    EncodedNumber** recv_encrypted_statistics;
    deserialize_encrypted_statistics(client_id, node_index, split_num, classes_num,
            recv_left_sample_nums, recv_right_sample_nums, recv_encrypted_statistics, ds);

    is_success = true;
    if (recv_encrypted_statistics[0][1].exponent == encrypted_statistics_x[0][1].exponent
         && recv_encrypted_statistics[0][1].type == encrypted_statistics_x[0][1].type
         && mpz_cmp(recv_encrypted_statistics[0][1].n, encrypted_statistics_x[0][1].n) == 0
         && mpz_cmp(recv_encrypted_statistics[0][1].value, encrypted_statistics_x[0][1].value) == 0) {
        is_success = true;
    } else {
        is_success = false;
    }
    if (is_success) {
        total_cases_num += 1;
        passed_cases_num += 1;
        logger(logger_out, "test_pb_encrypted_statistics: succeed\n");
    } else {
        total_cases_num += 1;
        logger(logger_out, "test_pb_encrypted_statistics: failed\n");
    }
}

void test_pb_updated_info() {
    int source_client_id = 1;
    int best_client_id = 1;
    int best_feature_id = 2;
    int best_split_id = 3;
    int sample_size = 2;
    EncodedNumber left_impurity, right_impurity;
    left_impurity.set_float(n, 0.1);
    right_impurity.set_float(n, 0.2);
    EncodedNumber *left_sample_iv = new EncodedNumber[sample_size];
    EncodedNumber *right_sample_iv = new EncodedNumber[sample_size];

    left_sample_iv[0].set_float(n, 0.123456);
    left_sample_iv[0].type = Ciphertext;
    left_sample_iv[1].set_float(n, 0.222222);
    left_sample_iv[1].type = Ciphertext;
    right_sample_iv[0].set_float(n, 0.555555);
    right_sample_iv[0].type = Ciphertext;
    right_sample_iv[1].set_float(n, 0.888888);
    right_sample_iv[1].type = Ciphertext;

    std::string send_s;
    serialize_update_info(source_client_id, best_client_id, best_feature_id, best_split_id,
            left_impurity, right_impurity, left_sample_iv, right_sample_iv, sample_size, send_s);
    // deserialize
    int recv_source_client_id, recv_best_client_id, recv_best_feature_id, recv_best_split_id;
    EncodedNumber recv_left_impurity, recv_right_impurity;
    EncodedNumber *recv_left_sample_iv, *recv_right_sample_iv;
    deserialize_update_info(recv_source_client_id, recv_best_client_id, recv_best_feature_id, recv_best_split_id,
            recv_left_impurity, recv_right_impurity, recv_left_sample_iv, recv_right_sample_iv, send_s);

    bool is_success = true;
    is_success = is_success && (recv_left_sample_iv[0].exponent == left_sample_iv[0].exponent);
    is_success = is_success && (recv_left_sample_iv[0].type == left_sample_iv[0].type);
    is_success = is_success && (mpz_cmp(recv_left_sample_iv[0].n, left_sample_iv[0].n) == 0);
    is_success = is_success && (mpz_cmp(recv_left_sample_iv[0].value, left_sample_iv[0].value) == 0);
    is_success = is_success && ((recv_best_client_id == best_client_id) && (recv_best_feature_id == best_feature_id)
                    && (recv_best_split_id == best_split_id) && (recv_source_client_id == source_client_id));
    if (is_success) {
        total_cases_num += 1;
        passed_cases_num += 1;
        logger(logger_out, "test_pb_updated_info: succeed\n");
    } else {
        total_cases_num += 1;
        logger(logger_out, "test_pb_updated_info: failed\n");
    }
}

void test_pb_split_info() {
    int global_split_num = 5;
    std::vector<int> split_nums;
    split_nums.push_back(1);
    split_nums.push_back(2);
    std::string s;
    serialize_split_info(global_split_num, split_nums, s);
    // deserialize
    int recv_global_split_num;
    std::vector<int> recv_split_nums;
    deserialize_split_info(recv_global_split_num, recv_split_nums, s);
    bool is_success = true;
    if ((recv_global_split_num == global_split_num) && (recv_split_nums[0] == split_nums[0]) && (recv_split_nums[1] == split_nums[1])) {
        is_success = true;
    } else {
        is_success = false;
    }
    if (is_success) {
        total_cases_num += 1;
        passed_cases_num += 1;
        logger(logger_out, "test_pb_split_info: succeed\n");
    } else {
        total_cases_num += 1;
        logger(logger_out, "test_pb_split_info: failed\n");
    }
}

void test_pb_prune_check_result() {
    int node_index = 1;
    int is_satisfied = 1;
    EncodedNumber label;
    label.set_float(n, 1.0);
    label.type = Plaintext;
    std::string s;
    serialize_prune_check_result(node_index, is_satisfied, label, s);
    int recv_node_index, recv_is_satisfied;
    EncodedNumber recv_label;
    deserialize_prune_check_result(recv_node_index, recv_is_satisfied, recv_label, s);
    // test equals
    bool is_success = true;
    if ((recv_node_index == node_index) && (recv_is_satisfied == is_satisfied)
        && label.exponent == recv_label.exponent
        && label.type == recv_label.type
        && mpz_cmp(label.n, recv_label.n) == 0
        && mpz_cmp(label.value, recv_label.value) == 0) {
        is_success = true;
    } else {
        is_success = false;
    }
    if (is_success) {
        total_cases_num += 1;
        passed_cases_num += 1;
        logger(logger_out, "test_pb_prune_check_result: succeed\n");
    } else {
        total_cases_num += 1;
        logger(logger_out, "test_pb_prune_check_result: failed\n");
    }
}

void test_pb_encrypted_label_vector() {
    EncodedNumber *encrypted_label_vector = new EncodedNumber[4];
    encrypted_label_vector[0].set_float(n, 0.123456);
    encrypted_label_vector[0].type = Ciphertext;
    encrypted_label_vector[1].set_float(n, 0.654321);
    encrypted_label_vector[1].type = Ciphertext;
    encrypted_label_vector[2].set_float(n, 0.123456);
    encrypted_label_vector[2].type = Ciphertext;
    encrypted_label_vector[3].set_float(n, 0.654321);
    encrypted_label_vector[3].type = Ciphertext;
    int node_index = 1;
    std::string s;
    serialize_encrypted_label_vector(node_index, 2, 2, encrypted_label_vector, s);
    int recv_node_index;
    EncodedNumber *deserialized_label_vector = new EncodedNumber[4];
    deserialize_encrypted_label_vector(recv_node_index, deserialized_label_vector, s);
    // test equals
    bool is_success = true;
    for (int i = 0; i < 4; i++) {
        if (encrypted_label_vector[i].exponent == deserialized_label_vector[i].exponent
            && encrypted_label_vector[i].type == deserialized_label_vector[i].type
            && mpz_cmp(encrypted_label_vector[i].n, deserialized_label_vector[i].n) == 0
            && mpz_cmp(encrypted_label_vector[i].value, deserialized_label_vector[i].value) == 0) {
            continue;
        } else {
            is_success = false;
        }
    }
    if (is_success) {
        total_cases_num += 1;
        passed_cases_num += 1;
        logger(logger_out, "test_pb_encrypted_label_vector: succeed\n");
    } else {
        total_cases_num += 1;
        logger(logger_out, "test_pb_encrypted_label_vector: failed\n");
    }
}

int test_pb() {
    logger(logger_out, "****** Test protobuf serialization and deserialization ******\n");
    total_cases_num = 0;
    passed_cases_num = 0;
    test_pb_encode_number();
    test_pb_batch_ids();
    test_pb_batch_sums();
    test_pb_batch_losses();
    test_pb_pruning_condition_result();
    test_pb_encrypted_statistics();
    test_pb_updated_info();
    test_pb_split_info();
    test_pb_prune_check_result();
    test_pb_encrypted_label_vector();
    logger(logger_out, "****** total_cases_num = %d, passed_cases_num = %d ******\n",
           total_cases_num, passed_cases_num);
    return 0;
}

