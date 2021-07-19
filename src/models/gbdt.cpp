//
// Created by wuyuncheng on 12/1/20.
//

#include "gbdt.h"
#include "../utils/djcs_t_aux.h"
#include "../utils/pb_converter.h"
#include <iomanip>
#include <random>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/inotify.h>
#include <map>
#include <stack>
#include "../utils/score.h"
#include "../utils/spdz/spdz_util.h"

extern FILE * logger_out;

GBDT::GBDT() {}

GBDT::GBDT(int m_tree_num, int m_global_feature_num,
    int m_local_feature_num, int m_internal_node_num, int m_type,
    int m_classes_num, int m_max_depth, int m_max_bins,
    int m_prune_sample_num, float m_prune_threshold,
    int solution_type, int optimization_type) {
    num_trees = m_tree_num;
    gbdt_type = m_type;
    if (gbdt_type == Regression) {
        classes_num = 1;
    } else {
        classes_num = m_classes_num;
    }
    learning_rates.reserve(num_trees);
    for (int i = 0; i < num_trees; i++) {
        learning_rates.emplace_back(GBDT_LEARNING_RATE);
    }
    forest_size = classes_num * num_trees;
    forest.reserve(forest_size);
    for (int i = 0; i < forest_size; ++i) {
        forest.emplace_back(m_global_feature_num, m_local_feature_num,
            m_internal_node_num, 1, m_classes_num,
            m_max_depth, m_max_bins, m_prune_sample_num,
            m_prune_threshold, solution_type, optimization_type);
    }
    logger(logger_out, "GBDT_type = %d, init %d trees in the GBDT\n", gbdt_type, forest_size);
}

void GBDT::init_datasets(Client & client, float split) {
    logger(logger_out, "Begin init dataset\n");
    int training_data_size = client.sample_num * split;
    // store the indexes of the training dataset for random batch selection
    std::vector<int> data_indexes;
    for (int i = 0; i < client.sample_num; i++) {
        data_indexes.push_back(i);
    }
    std::random_device rd;
    std::default_random_engine rng(rd());
    //auto rng = std::default_random_engine();
    std::shuffle(std::begin(data_indexes), std::end(data_indexes), rng);
    // select the former training data size as training data, and the latter as testing data
    for (int i = 0; i < data_indexes.size(); i++) {
        if (i < training_data_size) {
            // add to training dataset and labels
            training_data.push_back(client.local_data[data_indexes[i]]);
            if (client.has_label) {
                training_data_labels.push_back(client.labels[data_indexes[i]]);
            }
        } else {
            // add to testing dataset and labels
            testing_data.push_back(client.local_data[data_indexes[i]]);
            if (client.has_label) {
                testing_data_labels.push_back(client.labels[data_indexes[i]]);
            }
        }
    }
    int *new_indexes = new int[client.sample_num];
    for (int i = 0; i < client.sample_num; i++) {
        new_indexes[i] = data_indexes[i];
    }

    // send the data_indexes to the other client, and the other client splits in the same way
    for (int i = 0; i < client.client_num; i++) {
        if (i != client.client_id) {
            std::string s;
            serialize_batch_ids(new_indexes, client.sample_num, s);
            client.send_long_messages(i, s);
        }
    }
    logger(logger_out, "End init dataset\n");
    delete [] new_indexes;
}

void GBDT::init_datasets_with_indexes(Client & client, int new_indexes[], float split) {
    logger(logger_out, "Begin init dataset with indexes\n");
    int training_data_size = client.sample_num * split;
    // select the former training data size as training data, and the latter as testing data
    for (int i = 0; i < client.sample_num; i++) {
        if (i < training_data_size) {
            // add to training dataset and labels
            training_data.push_back(client.local_data[new_indexes[i]]);
            if (client.has_label) {
                training_data_labels.push_back(client.labels[new_indexes[i]]);
            }
        } else {
            // add to testing dataset and labels
            testing_data.push_back(client.local_data[new_indexes[i]]);
            if (client.has_label) {
                testing_data_labels.push_back(client.labels[new_indexes[i]]);
            }
        }
    }
    logger(logger_out, "End init dataset with indexes\n");
}

void GBDT::init_single_tree_data(Client &client, int class_id,
    int tree_id, std::vector<float> cur_predicted_labels) {
    int real_tree_id = class_id * num_trees + tree_id;
    forest[real_tree_id].training_data = training_data;
    forest[real_tree_id].testing_data = testing_data;
    forest[real_tree_id].classes_num = 2;  // for regression, the classes num is set to 2 for y and y^2
    if (client.client_id == SUPER_CLIENT_ID) {
        if (tree_id == 0) { // just copy the original labels
            if (gbdt_type == Regression) {
                forest[real_tree_id].training_data_labels = training_data_labels;
            } else {
                // one-hot label encoder
                for (int i = 0; i < training_data.size(); i++) {
                    if ((float) training_data_labels[i] == class_id) {
                        forest[real_tree_id].training_data_labels.push_back(1.0);
                    } else {
                        forest[real_tree_id].training_data_labels.push_back(0.0);
                    }
                }
            }
        } else { // should use the predicted labels of first tree
            for (int i = 0; i < training_data.size(); i++) {
                forest[real_tree_id].training_data_labels.push_back(
                        forest[class_id * num_trees].training_data_labels[i] - cur_predicted_labels[i]);
            }
        }
        // pre-compute indicator vectors or variance vectors for labels
        // here already assume that client_id == 0 (super client)
        // regression, compute variance necessary stats
        std::vector<float> label_square_vec;
        for (int j = 0; j < training_data_labels.size(); j++) {
            label_square_vec.push_back(forest[real_tree_id].training_data_labels[j] * forest[real_tree_id].training_data_labels[j]);
        }
        // the first vector is the actual label vector
        forest[real_tree_id].variance_stat_vecs.push_back(forest[real_tree_id].training_data_labels);
        // the second vector is the squared label vector
        forest[real_tree_id].variance_stat_vecs.push_back(label_square_vec);
    }

}

void GBDT::build_gbdt(Client &client) {
    /**
     * 1. For regression, build as follows:
     *  (1) from tree 0 to tree max, init a decision tree; if client id == 0, init with difference label
     *  (2) build a decision tree using building blocks in cart_tree.h
     *  (3) after building the current tree, compute the predicted labels
     *      for the current training dataset for the next tree
     *
     * 2. For classification, build as follows:
     *  (1) for each class, convert to the one-hot encoding dataset,
     *      init classes_num forests, each forest init the first tree
     *  (2) from tree 0 to tree max, build iteratively using building blocks in cart_tree.h
     *  (3) after building trees in the current iteration, compute the predicted distribution
     *      for the training dataset, and compute the losses for init the difference of
     *      training labels in the trees of the next iteration
     */

    logger(logger_out, "Begin to build GBDT model\n");
    // this vector is to store the predicted labels of current iteration
    std::vector< std::vector<float> > cur_predicted_labels;
    for (int class_id = 0; class_id < classes_num; class_id++) {
        std::vector<float> t;
        for (int i = 0; i < training_data.size(); i++) {
            t.push_back(0.0);
        }
        cur_predicted_labels.push_back(t);
    }
    // build trees iteratively
    for (int tree_id = 0; tree_id < num_trees; tree_id++) {
        logger(logger_out, "------------------- build the %d-th tree ----------------------\n", tree_id);
        std::vector< std::vector<float> > softmax_predicted_labels;
        for (int class_id = 0; class_id < classes_num; class_id++) {
            std::vector<float> t;
            for (int i = 0; i < training_data.size(); i++) {
                t.push_back(0.0);
            }
            softmax_predicted_labels.push_back(t);
        }
        logger(logger_out, "Init softmax labels\n");
        if (gbdt_type == Classification && tree_id != 0) {
            // compute predicted labels for classification
            for (int i = 0; i < training_data.size(); i++) {
                std::vector<float> prob_distribution_i;
                for (int class_id = 0; class_id < classes_num; class_id++) {
                    prob_distribution_i.push_back(cur_predicted_labels[class_id][i]);
                }
                std::vector<float> softmax_prob_distribution_i = softmax(prob_distribution_i);
                for (int class_id = 0; class_id < classes_num; class_id++) {
                    softmax_predicted_labels[class_id][i] = softmax_prob_distribution_i[class_id];
                }
            }
        }
        for (int class_id = 0; class_id < classes_num; class_id++) {
            // init single tree dataset
            if (gbdt_type == Classification) {
                init_single_tree_data(client, class_id, tree_id, softmax_predicted_labels[class_id]);
            } else {
                init_single_tree_data(client, class_id, tree_id, cur_predicted_labels[class_id]);
            }
            //build the current tree
            int real_tree_id = class_id * num_trees + tree_id;
            forest[real_tree_id].init_features();
            forest[real_tree_id].init_root_node(client);
            forest[real_tree_id].build_tree_node(client, 0);
            // after the tree has been built, compute the predicted labels for the current tree
            std::vector<float> predicted_training_labels = compute_predicted_labels(client,
                class_id, tree_id, 0);
            for (int i = 0; i < training_data.size(); i++) {
                cur_predicted_labels[class_id][i] += GBDT_LEARNING_RATE * predicted_training_labels[i];
            }
            forest[real_tree_id].intermediate_memory_free();
        }
    }

    logger(logger_out, "End to build GBDT model\n");
}

std::vector<float> GBDT::compute_predicted_labels(Client &client,
    int class_id, int tree_id, int flag) {
    int real_tree_id = class_id * num_trees + tree_id;
    std::vector<float> predicted_label_vector;
    std::vector< std::vector<float> > input_dataset;
    int size = 0;
    if (flag == 0) { // training dataset
        input_dataset = training_data;
        size = training_data.size();
    } else {
        input_dataset = testing_data;
        size = testing_data.size();
    }
    for (int i = 0; i < size; i++) {
        predicted_label_vector.push_back(0.0);
    }
    // for each sample
    for (int i = 0; i < input_dataset.size(); ++i) {
        // step 1: organize the leaf label vector, compute the map
        EncodedNumber *label_vector = new EncodedNumber[forest[real_tree_id].internal_node_num + 1];
        std::map<int, int> node_index_2_leaf_index_map;
        int leaf_cur_index = 0;
        for (int j = 0; j < pow(2, forest[real_tree_id].max_depth + 1) - 1; j++) {
            if (forest[real_tree_id].tree_nodes[j].is_leaf == 1) {
                node_index_2_leaf_index_map.insert(std::make_pair(j, leaf_cur_index));
                label_vector[leaf_cur_index] = forest[real_tree_id].tree_nodes[j].label;  // record leaf label vector
                leaf_cur_index++;
            }
        }
        // compute binary vector for the current sample
        std::vector<float> sample_values = input_dataset[i];
        std::vector<int> binary_vector = compute_binary_vector(class_id,
            tree_id, sample_values, node_index_2_leaf_index_map);
        EncodedNumber *encoded_binary_vector = new EncodedNumber[binary_vector.size()];
        EncodedNumber *updated_label_vector;// = new EncodedNumber[binary_vector.size()];
        // update in Robin cycle, from the last client to client 0
        if (client.client_id == client.client_num - 1) {
            updated_label_vector = new EncodedNumber[binary_vector.size()];
            for (int j = 0; j < binary_vector.size(); j++) {
                encoded_binary_vector[j].set_integer(client.m_pk->n[0], binary_vector[j]);
                djcs_t_aux_ep_mul(client.m_pk, updated_label_vector[j],
                    label_vector[j], encoded_binary_vector[j]);
            }
            // send to the next client
            std::string send_s;
            serialize_batch_sums(updated_label_vector, binary_vector.size(), send_s);
            client.send_long_messages(client.client_id - 1, send_s);
        } else if (client.client_id > 0) {
            std::string recv_s;
            client.recv_long_messages(client.client_id + 1, recv_s);
            int recv_size; // should be same as binary_vector.size()
            deserialize_sums_from_string(updated_label_vector, recv_size, recv_s);
            for (int j = 0; j < binary_vector.size(); j++) {
                encoded_binary_vector[j].set_integer(client.m_pk->n[0], binary_vector[j]);
                djcs_t_aux_ep_mul(client.m_pk, updated_label_vector[j],
                    updated_label_vector[j], encoded_binary_vector[j]);
            }
            std::string resend_s;
            serialize_batch_sums(updated_label_vector, binary_vector.size(), resend_s);
            client.send_long_messages(client.client_id - 1, resend_s);
        } else {
            // the super client update the last, and aggregate before calling share decryption
            std::string final_recv_s;
            client.recv_long_messages(client.client_id + 1, final_recv_s);
            int final_recv_size;
            deserialize_sums_from_string(updated_label_vector, final_recv_size, final_recv_s);
            for (int j = 0; j < binary_vector.size(); j++) {
                encoded_binary_vector[j].set_integer(client.m_pk->n[0], binary_vector[j]);
                djcs_t_aux_ep_mul(client.m_pk, updated_label_vector[j],
                    updated_label_vector[j], encoded_binary_vector[j]);
            }
        }
        // aggregate and call share decryption
        if (client.client_id == SUPER_CLIENT_ID) {
            EncodedNumber *encrypted_aggregation = new EncodedNumber[1];
            encrypted_aggregation[0].set_float(client.m_pk->n[0], 0, 2 * FLOAT_PRECISION);
            djcs_t_aux_encrypt(client.m_pk, client.m_hr,
                encrypted_aggregation[0], encrypted_aggregation[0]);
            for (int j = 0; j < binary_vector.size(); j++) {
                djcs_t_aux_ee_add(client.m_pk, encrypted_aggregation[0],
                    encrypted_aggregation[0], updated_label_vector[j]);
            }
            float label;
            EncodedNumber *decrypted_label = new EncodedNumber[1];
            client.share_batch_decrypt(encrypted_aggregation, decrypted_label, 1);
            decrypted_label[0].decode(label);
            predicted_label_vector[i] = label;
            delete [] encrypted_aggregation;
            delete [] decrypted_label;
        } else {
            std::string s, response_s;
            client.recv_long_messages(SUPER_CLIENT_ID, s);
            client.decrypt_batch_piece(s, response_s, SUPER_CLIENT_ID);
        }
        delete[] encoded_binary_vector;
        delete[] updated_label_vector;
        delete[] label_vector;
    }
    return predicted_label_vector;
}

std::vector<int> GBDT::compute_binary_vector(int class_id, int tree_id, std::vector<float> sample_values,
                                             std::map<int, int> node_index_2_leaf_index_map) {
    int real_tree_id = class_id * num_trees + tree_id;
    std::vector<int> binary_vector(forest[real_tree_id].internal_node_num + 1);
    // traverse the whole tree iteratively, and compute binary_vector
    std::stack<PredictionObj> traverse_prediction_objs;
    PredictionObj prediction_obj(forest[real_tree_id].tree_nodes[0].is_leaf,
        forest[real_tree_id].tree_nodes[0].is_self_feature,
        forest[real_tree_id].tree_nodes[0].best_client_id,
        forest[real_tree_id].tree_nodes[0].best_feature_id,
        forest[real_tree_id].tree_nodes[0].best_split_id, 1, 0);
    traverse_prediction_objs.push(prediction_obj);
    while (!traverse_prediction_objs.empty()) {
        PredictionObj pred_obj = traverse_prediction_objs.top();
        if (pred_obj.is_leaf == 1) {
            // find leaf index and record
            int leaf_index = node_index_2_leaf_index_map.find(pred_obj.index)->second;
            binary_vector[leaf_index] = pred_obj.mark;
            traverse_prediction_objs.pop();
        } else if (pred_obj.is_self_feature != 1) {
            // both left and right branches are marked as 1 * current_mark
            traverse_prediction_objs.pop();
            int left_node_index = pred_obj.index * 2 + 1;
            int right_node_index = pred_obj.index * 2 + 2;
            PredictionObj left(forest[real_tree_id].tree_nodes[left_node_index].is_leaf,
                forest[real_tree_id].tree_nodes[left_node_index].is_self_feature,
                forest[real_tree_id].tree_nodes[left_node_index].best_client_id,
                forest[real_tree_id].tree_nodes[left_node_index].best_feature_id,
                forest[real_tree_id].tree_nodes[left_node_index].best_split_id,
                pred_obj.mark, left_node_index);
            PredictionObj right(forest[real_tree_id].tree_nodes[right_node_index].is_leaf,
                forest[real_tree_id].tree_nodes[right_node_index].is_self_feature,
                forest[real_tree_id].tree_nodes[right_node_index].best_client_id,
                forest[real_tree_id].tree_nodes[right_node_index].best_feature_id,
                forest[real_tree_id].tree_nodes[right_node_index].best_split_id,
                pred_obj.mark, right_node_index);
            traverse_prediction_objs.push(left);
            traverse_prediction_objs.push(right);
        } else {
            // is self feature, retrieve split value and compare
            traverse_prediction_objs.pop();
            int feature_id = pred_obj.best_feature_id;
            int split_id = pred_obj.best_split_id;
            float split_value = forest[real_tree_id].features[feature_id].split_values[split_id];
            int left_mark, right_mark;
            if (sample_values[feature_id] <= split_value) {
                left_mark = pred_obj.mark * 1;
                right_mark = pred_obj.mark * 0;
            } else {
                left_mark = pred_obj.mark * 0;
                right_mark = pred_obj.mark * 1;
            }
            int left_node_index = pred_obj.index * 2 + 1;
            int right_node_index = pred_obj.index * 2 + 2;
            PredictionObj left(forest[real_tree_id].tree_nodes[left_node_index].is_leaf,
                forest[real_tree_id].tree_nodes[left_node_index].is_self_feature,
                forest[real_tree_id].tree_nodes[left_node_index].best_client_id,
                forest[real_tree_id].tree_nodes[left_node_index].best_feature_id,
                forest[real_tree_id].tree_nodes[left_node_index].best_split_id,
                left_mark, left_node_index);
            PredictionObj right(forest[real_tree_id].tree_nodes[right_node_index].is_leaf,
                forest[real_tree_id].tree_nodes[right_node_index].is_self_feature,
                forest[real_tree_id].tree_nodes[right_node_index].best_client_id,
                forest[real_tree_id].tree_nodes[right_node_index].best_feature_id,
                forest[real_tree_id].tree_nodes[right_node_index].best_split_id,
                right_mark, right_node_index);
            traverse_prediction_objs.push(left);
            traverse_prediction_objs.push(right);
        }
    }
    return binary_vector;
}

void GBDT::build_gbdt_with_spdz(Client &client) {
    /**
     * 1. For regression, build as follows:
     *  (1) from tree 0 to tree max, init a decision tree; if client id == 0, init with encrypted difference label
     *  (2) build a decision tree using building blocks in cart_tree.h (modify the init tree node function)
     *  (3) after building the current tree, compute the encrypted predicted labels for the current training dataset for the next tree
     *
     * 2. For classification, build as follows:
     *  (1) for each class, convert to the one-hot encoding dataset, init classes_num forests, each forest init the first tree
     *  (2) from tree 0 to tree max, build iteratively using building blocks in cart_tree.h
     *  (3) after building trees in the current iteration, compute the predicted distribution for the training dataset, and compute
     *      the losses for init the difference of training labels in the trees of the next iteration
     */

    logger(logger_out, "Begin to build GBDT model\n");
    int sample_num = training_data.size();
    EncodedNumber * cur_predicted_labels = new EncodedNumber[classes_num * sample_num];
    for (int i = 0; i < classes_num; i++) {
        for (int j = 0; j < sample_num; j++) {
            cur_predicted_labels[i * sample_num + j].set_float(client.m_pk->n[0], 0.0);
            djcs_t_aux_encrypt(client.m_pk, client.m_hr,
                cur_predicted_labels[i * sample_num + j], cur_predicted_labels[i * sample_num + j]);
        }
    }

    // build trees iteratively
    for (int tree_id = 0; tree_id < num_trees; tree_id++) {
        logger(logger_out, "------------------- build the %d-th tree ----------------------\n", tree_id);
        if (gbdt_type == Regression) {
            // 1. init single tree data except the labels
            init_simplified_single_tree_data(client, 0, tree_id);
            int real_tree_id = 0 * num_trees + tree_id;
            forest[real_tree_id].init_features();
            // 2. construct encrypted labels (skip variance_stat_vecs, directly to init root node)
            EncodedNumber * encrypted_label_vector = new EncodedNumber[sample_num];
            if (client.client_id == SUPER_CLIENT_ID) {
                logger(logger_out, "the first training data label is = %f\n", forest[0].training_data_labels[0]);
                for (int i = 0; i < sample_num; i++) {
                    encrypted_label_vector[i].set_float(client.m_pk->n[0], forest[0].training_data_labels[i]);
                    djcs_t_aux_encrypt(client.m_pk, client.m_hr,
                        encrypted_label_vector[i], encrypted_label_vector[i]);
                }
                if (tree_id != 0) {
                    // compute encrypted difference between original labels and the cur_predicted_labels
                    EncodedNumber * helper_cur_predicted_labels = new EncodedNumber[sample_num];
                    EncodedNumber constant;
                    constant.set_integer(client.m_pk->n[0], -1);
                    for (int i = 0; i < sample_num; i++) {
                        djcs_t_aux_ep_mul(client.m_pk, helper_cur_predicted_labels[i],
                            cur_predicted_labels[i], constant);
                        djcs_t_aux_ee_add(client.m_pk, encrypted_label_vector[i],
                            encrypted_label_vector[i], helper_cur_predicted_labels[i]);
                    }
                    delete [] helper_cur_predicted_labels;
                }
            }
            // 3. compute squared label encrypted vector, call spdz
            EncodedNumber * encrypted_square_label_vector;
            compute_squared_label_vector(client, encrypted_square_label_vector, encrypted_label_vector);
            // 4. init root node
            init_root_node_gbdt(client, real_tree_id, encrypted_label_vector, encrypted_square_label_vector);
            // 5. train a single tree
            forest[real_tree_id].build_tree_node(client, 0);
            // 6. compute the predicted labels and update cur_predicted_labels
            // after the tree has been built, compute the predicted labels for the current tree
            EncodedNumber * encrypted_predicted_labels;
            compute_encrypted_predicted_labels(client, 0,
                tree_id, encrypted_predicted_labels, 0);
            if (client.client_id == SUPER_CLIENT_ID) {
                for (int i = 0; i < sample_num; i++) {
                    djcs_t_aux_ee_add(client.m_pk, cur_predicted_labels[i],
                        cur_predicted_labels[i], encrypted_predicted_labels[i]);
                }
            }
            forest[real_tree_id].intermediate_memory_free();

            delete [] encrypted_label_vector;
            delete [] encrypted_square_label_vector;
            delete [] encrypted_predicted_labels;
        } else {
            // compute softmax labels
            EncodedNumber * res_softmax_label_vector = new EncodedNumber[sample_num * classes_num];
            if (tree_id != 0) {
                compute_softmax_label_vector(client, res_softmax_label_vector, cur_predicted_labels);
            }
            for (int class_id = 0; class_id < classes_num; class_id++) {
                // 1. init single tree data except the labels
                init_simplified_single_tree_data(client, class_id, tree_id);
                int real_tree_id = class_id * num_trees + tree_id;
                forest[real_tree_id].init_features();
                // 2. construct encrypted softmax labels for classification
                EncodedNumber * encrypted_label_vector = new EncodedNumber[sample_num];
                if (client.client_id == SUPER_CLIENT_ID) {
                    for (int i = 0; i < sample_num; i++) {
                        encrypted_label_vector[i].set_float(client.m_pk->n[0],
                            forest[class_id * num_trees].training_data_labels[i]);
                        djcs_t_aux_encrypt(client.m_pk, client.m_hr,
                            encrypted_label_vector[i], encrypted_label_vector[i]);
                    }
                    if (tree_id != 0) {
                        // compute encrypted difference between original labels and the cur_predicted_labels
                        EncodedNumber * helper_cur_predicted_labels = new EncodedNumber[sample_num];
                        EncodedNumber constant;
                        constant.set_integer(client.m_pk->n[0], -1);
                        for (int i = 0; i < sample_num; i++) {
                            djcs_t_aux_ep_mul(client.m_pk, helper_cur_predicted_labels[i],
                                    res_softmax_label_vector[class_id * sample_num + i], constant);
                            djcs_t_aux_ee_add(client.m_pk, encrypted_label_vector[i],
                                    encrypted_label_vector[i], helper_cur_predicted_labels[i]);
                        }
                        delete [] helper_cur_predicted_labels;
                    }
                }
                // 3. compute squared label encrypted vector
                EncodedNumber * encrypted_square_label_vector = new EncodedNumber[sample_num];
                compute_squared_label_vector(client, encrypted_square_label_vector, encrypted_label_vector);
                // 4. init root node
                init_root_node_gbdt(client, real_tree_id, encrypted_label_vector, encrypted_square_label_vector);
                // 5. train a single tree
                forest[real_tree_id].build_tree_node(client, 0);
                // 6. compute the predicted labels and update cur_predicted_labels
                // after the tree has been built, compute the predicted labels for the current tree
                EncodedNumber * encrypted_predicted_labels = new EncodedNumber[sample_num];
                compute_encrypted_predicted_labels(client, class_id, tree_id, encrypted_predicted_labels, 0);
                if (client.client_id == SUPER_CLIENT_ID) {
                    for (int i = 0; i < sample_num; i++) {
                        djcs_t_aux_ee_add(client.m_pk, cur_predicted_labels[class_id * sample_num + i],
                            cur_predicted_labels[class_id * sample_num + i], encrypted_predicted_labels[i]);
                    }
                }
                forest[real_tree_id].intermediate_memory_free();
                delete [] encrypted_label_vector;
                delete [] encrypted_square_label_vector;
                delete [] encrypted_predicted_labels;
            }
            delete [] res_softmax_label_vector;
        }
    }
    delete [] cur_predicted_labels;
}

void GBDT::init_simplified_single_tree_data(Client &client, int class_id, int tree_id) {
    int real_tree_id = class_id * num_trees + tree_id;
    forest[real_tree_id].training_data = training_data;
    forest[real_tree_id].testing_data = testing_data;
    forest[real_tree_id].classes_num = 2;  // for regression, the classes num is set to 2 for y and y^2
    if (client.client_id == SUPER_CLIENT_ID) {
        if (tree_id == 0) { // just copy the original labels
            if (gbdt_type == Regression) {
                forest[real_tree_id].training_data_labels = training_data_labels;
            } else {
                // one-hot label encoder
                for (int i = 0; i < training_data.size(); i++) {
                    if ((float) training_data_labels[i] == class_id) {
                        forest[real_tree_id].training_data_labels.push_back(1.0);
                    } else {
                        forest[real_tree_id].training_data_labels.push_back(0.0);
                    }
                }
            }
        }
    }
}

void GBDT::compute_squared_label_vector(Client &client,
    EncodedNumber *&squared_label_vector,
    EncodedNumber *encrypted_label_vector) {
    int sample_num = training_data.size();
    std::vector<float> label_vector_shares;
    squared_label_vector = new EncodedNumber[sample_num];
    /*** init static gfp for sending private batch shares and setup sockets ***/
    string prep_data_prefix = get_prep_dir(NUM_SPDZ_PARTIES, SPDZ_LG2P, gf2n::default_degree());
    initialise_fields(prep_data_prefix);
    bigint::init_thread();
    std::vector<int> sockets = setup_sockets(NUM_SPDZ_PARTIES,
        client.client_id, client.host_names, SPDZ_PORT_NUM_DT);
    if (client.client_id == SUPER_CLIENT_ID) {
        // the super client sends computation id for SPDZ computation of a specific branch
        std::vector<int> computation_id;
        computation_id.push_back(GBDTLabelSquare);
        send_public_values(computation_id, sockets, NUM_SPDZ_PARTIES);
        std::vector<int> parameters;
        parameters.push_back(sample_num);
        parameters.push_back(classes_num);
        send_public_values(parameters, sockets, NUM_SPDZ_PARTIES);
        logger(logger_out, "sample_size = %d, classes_num = %d\n", sample_num, classes_num);
        // convert the encrypted label vector into secret shares
        client.ciphers_conversion_to_shares(encrypted_label_vector,
            label_vector_shares, sample_num);
    } else {
        client.ciphers_conversion_to_shares(encrypted_label_vector,
            label_vector_shares, sample_num);
    }
    logger(logger_out, "label_vector_shares[0] = %f\n", label_vector_shares[0]);
    // send shares to spdz parties
    for (int i = 0; i < sample_num; i++) {
        vector<float> x;
        x.push_back(label_vector_shares[i]);
        send_private_batch_shares(x, sockets, NUM_SPDZ_PARTIES);
    }
    // receive square label vector shares from spdz parties
    std::vector<float> square_label_vector_shares = receive_result(sockets, NUM_SPDZ_PARTIES, sample_num);
    logger(logger_out, "square_label_vector_shares[0] = %f\n", square_label_vector_shares[0]);
    // close connection with the SPDZ parties, otherwise, the next node cannot connect
    for (unsigned int i = 0; i < sockets.size(); i++) {
        close_client_socket(sockets[i]);
    }
    // construct encrypted share vector
    for (int i = 0; i < sample_num; i++) {
        squared_label_vector[i].set_float(client.m_pk->n[0], square_label_vector_shares[i]);
        djcs_t_aux_encrypt(client.m_pk, client.m_hr,
            squared_label_vector[i], squared_label_vector[i]);
    }
    // aggregate the shares to encrypted_square_label_vector
    if (client.client_id == SUPER_CLIENT_ID) {
        for (int cid = 0; cid < client.client_num; cid++) {
            if (cid != client.client_id) {
                // receive from the other client and aggregate
                std::string recv_square_label_vector_str;
                client.recv_long_messages(cid, recv_square_label_vector_str);
                EncodedNumber * recv_square_label_vector = new EncodedNumber[sample_num];
                deserialize_sums_from_string(recv_square_label_vector,
                    sample_num, recv_square_label_vector_str);
                for (int i = 0; i < sample_num; i++) {
                    djcs_t_aux_ee_add(client.m_pk, squared_label_vector[i],
                        squared_label_vector[i], recv_square_label_vector[i]);
                }
                delete [] recv_square_label_vector;
            }
        }
    } else {
        // serialize to send to the super client
        std::string send_square_label_vector_str;
        serialize_batch_sums(squared_label_vector, sample_num, send_square_label_vector_str);
        client.send_long_messages(SUPER_CLIENT_ID, send_square_label_vector_str);
    }
    logger(logger_out, "Finish computing encrypted square label vector\n");
}

void GBDT::compute_softmax_label_vector(Client &client, EncodedNumber *&softmax_label_vector,
                                        EncodedNumber *encrypted_classes_label_vector) {
    int sample_num = training_data.size();
    std::vector<float> label_vector_shares;
    /*** init static gfp for sending private batch shares and setup sockets ***/
    string prep_data_prefix = get_prep_dir(NUM_SPDZ_PARTIES, SPDZ_LG2P, gf2n::default_degree());
    initialise_fields(prep_data_prefix);
    bigint::init_thread();
    std::vector<int> sockets = setup_sockets(NUM_SPDZ_PARTIES,
        client.client_id, client.host_names, SPDZ_PORT_NUM_DT);
    if (client.client_id == SUPER_CLIENT_ID) {
        // the super client sends computation id for SPDZ computation of a specific branch
        std::vector<int> computation_id;
        computation_id.push_back(GBDTSoftmax);
        send_public_values(computation_id, sockets, NUM_SPDZ_PARTIES);
        std::vector<int> parameters;
        parameters.push_back(sample_num);
        parameters.push_back(classes_num);
        send_public_values(parameters, sockets, NUM_SPDZ_PARTIES);
        logger(logger_out, "sample_size = %d, classes_num = %d\n", sample_num, classes_num);
        // convert the encrypted label vector into secret shares
        client.ciphers_conversion_to_shares(encrypted_classes_label_vector,
            label_vector_shares, sample_num * classes_num);
    } else {
        client.ciphers_conversion_to_shares(encrypted_classes_label_vector,
            label_vector_shares, sample_num * classes_num);
    }
    logger(logger_out, "label_vector_shares[0] = %f\n", label_vector_shares[0]);
    logger(logger_out, "label_vector_shares[%d] = %f\n", sample_num, label_vector_shares[sample_num]);

    // send shares to spdz parties
    for (int i = 0; i < sample_num * classes_num; i++) {
        vector<float> x;
        x.push_back(label_vector_shares[i]);
        send_private_batch_shares(x, sockets, NUM_SPDZ_PARTIES);
    }
    // receive square label vector shares from spdz parties
    std::vector<float> softmax_label_vector_shares = receive_result(sockets,
        NUM_SPDZ_PARTIES, sample_num * classes_num);
    logger(logger_out, "softmax_label_vector_shares[0] = %f\n", softmax_label_vector_shares[0]);
    logger(logger_out, "softmax_label_vector_shares[%d] = %f\n", sample_num, softmax_label_vector_shares[sample_num]);
    // close connection with the SPDZ parties, otherwise, the next node cannot connect
    for (unsigned int i = 0; i < sockets.size(); i++) {
        close_client_socket(sockets[i]);
    }
    // construct encrypted share vector
    for (int i = 0; i < sample_num * classes_num; i++) {
        softmax_label_vector[i].set_float(client.m_pk->n[0], softmax_label_vector_shares[i]);
        djcs_t_aux_encrypt(client.m_pk, client.m_hr,
            softmax_label_vector[i], softmax_label_vector[i]);
    }
    // aggregate the shares to encrypted_square_label_vector
    if (client.client_id == SUPER_CLIENT_ID) {
        for (int cid = 0; cid < client.client_num; cid++) {
            if (cid != client.client_id) {
                // receive from the other client and aggregate
                int total_size = sample_num * classes_num;
                std::string recv_softmax_label_vector_str;
                client.recv_long_messages(cid, recv_softmax_label_vector_str);
                EncodedNumber * recv_softmax_label_vector = new EncodedNumber[sample_num * classes_num];
                deserialize_sums_from_string(recv_softmax_label_vector,
                    total_size, recv_softmax_label_vector_str);
                for (int i = 0; i < sample_num * classes_num; i++) {
                    djcs_t_aux_ee_add(client.m_pk, softmax_label_vector[i],
                        softmax_label_vector[i], recv_softmax_label_vector[i]);
                }
                delete [] recv_softmax_label_vector;
            }
        }
    } else {
        // serialize to send to the super client
        std::string send_softmax_label_vector_str;
        serialize_batch_sums(softmax_label_vector,
            sample_num * classes_num, send_softmax_label_vector_str);
        client.send_long_messages(SUPER_CLIENT_ID, send_softmax_label_vector_str);
    }
    logger(logger_out, "Finish computing encrypted softmax label vector\n");
}

void GBDT::init_root_node_gbdt(Client &client, int real_tree_id, EncodedNumber *encrypted_label_vector,
                               EncodedNumber *encrypted_square_label_vector) {
    //logger(logger_out, "Begin init root node\n");
    // Note that for the root node, every client can init the encrypted sample mask vector
    // but the label vectors need to be received from the super client
    // assume that the global feature number is known beforehand
    int sample_num = training_data.size();
    forest[real_tree_id].tree_nodes[0].is_leaf = -1;
    forest[real_tree_id].tree_nodes[0].available_feature_ids.reserve(forest[real_tree_id].local_feature_num);
    for (int i = 0; i < forest[real_tree_id].local_feature_num; i++) {
        forest[real_tree_id].tree_nodes[0].available_feature_ids.push_back(i);
    }
    forest[real_tree_id].tree_nodes[0].available_global_feature_num = forest[real_tree_id].global_feature_num;
    forest[real_tree_id].tree_nodes[0].sample_size = sample_num;
    forest[real_tree_id].tree_nodes[0].classes_num = 2;
    forest[real_tree_id].tree_nodes[0].type = 1;
    forest[real_tree_id].tree_nodes[0].best_feature_id = -1;
    forest[real_tree_id].tree_nodes[0].best_client_id = -1;
    forest[real_tree_id].tree_nodes[0].best_split_id = -1;
    forest[real_tree_id].tree_nodes[0].depth = 0;
    forest[real_tree_id].tree_nodes[0].is_self_feature = -1;
    forest[real_tree_id].tree_nodes[0].left_child = -1;
    forest[real_tree_id].tree_nodes[0].right_child = -1;
    forest[real_tree_id].tree_nodes[0].sample_iv = new EncodedNumber[sample_num];
    forest[real_tree_id].tree_nodes[0].encrypted_labels = new EncodedNumber[2 * sample_num];

    // init encrypted mask vector on the root node
    EncodedNumber tmp;
    tmp.set_integer(client.m_pk->n[0], 1);
    for (int i = 0; i < training_data.size(); i++) {
        djcs_t_aux_encrypt(client.m_pk, client.m_hr, forest[real_tree_id].tree_nodes[0].sample_iv[i], tmp);
    }

    // if super client, compute the encrypted label information and broadcast to the other clients
    int used_classes_num = CLASS_NUM_FOR_REGRESSION; // default is regression tree
    if (client.client_id == SUPER_CLIENT_ID) {
        std::string result_str;
        // one dimension encrypted label vector
        EncodedNumber * encrypted_label_info = new EncodedNumber[used_classes_num * sample_num];
        for (int i = 0; i < sample_num; i++) {
            encrypted_label_info[i] = encrypted_label_vector[i];
            encrypted_label_info[sample_num + i] = encrypted_square_label_vector[i];
        }
        for (int i = 0; i < used_classes_num * sample_num; i++) {
            forest[real_tree_id].tree_nodes[0].encrypted_labels[i] = encrypted_label_info[i];
        }
        // serialize and send to the other client
        serialize_encrypted_label_vector(0, used_classes_num,
            sample_num, encrypted_label_info, result_str);
        for (int i = 0; i < client.client_num; i++) {
            if (i != client.client_id) {
                client.send_long_messages(i, result_str);
            }
        }
        delete [] encrypted_label_info;
    } else {
        // if not super client, receive the encrypted label information and set for the root node
        std::string recv_result_str;
        EncodedNumber * recv_encrypted_label_info;
        client.recv_long_messages(SUPER_CLIENT_ID, recv_result_str);
        int recv_node_index;
        deserialize_encrypted_label_vector(recv_node_index,
            recv_encrypted_label_info, recv_result_str);
        for (int i = 0; i < used_classes_num * sample_num; i++) {
            forest[real_tree_id].tree_nodes[0].encrypted_labels[i] = recv_encrypted_label_info[i];
        }
        delete [] recv_encrypted_label_info;
    }

    EncodedNumber max_variance;
    max_variance.set_float(client.m_pk->n[0], MAX_VARIANCE);
    djcs_t_aux_encrypt(client.m_pk, client.m_hr, forest[real_tree_id].tree_nodes[0].impurity, max_variance);
    logger(logger_out, "Init gbdt root node finished\n");
}

void GBDT::compute_encrypted_predicted_labels(Client &client, int class_id, int tree_id,
    EncodedNumber *&encrypted_predicted_labels, int flag) {
    int real_tree_id = class_id * num_trees + tree_id;
    std::vector< std::vector<float> > input_dataset;
    int size = 0;
    if (flag == 0) { // training dataset
        input_dataset = training_data;
        size = training_data.size();
    } else {
        input_dataset = testing_data;
        size = testing_data.size();
    }
    encrypted_predicted_labels = new EncodedNumber[size];
    // decrypt the label vector on leaf nodes
    // step 1: organize the leaf label vector, compute the map
    EncodedNumber *label_vector = new EncodedNumber[forest[real_tree_id].internal_node_num + 1];
    EncodedNumber *new_label_vector = new EncodedNumber[forest[real_tree_id].internal_node_num + 1];
    std::map<int, int> node_index_2_leaf_index_map;
    int leaf_cur_index = 0;
    for (int j = 0; j < pow(2, forest[real_tree_id].max_depth + 1) - 1; j++) {
        if (forest[real_tree_id].tree_nodes[j].is_leaf == 1) {
            node_index_2_leaf_index_map.insert(std::make_pair(j, leaf_cur_index));
            label_vector[leaf_cur_index] = forest[real_tree_id].tree_nodes[j].label;  // record leaf label vector
            leaf_cur_index++;
        }
    }
    if (client.client_id == client.client_num - 1) {
        EncodedNumber * decrypted_labels = new EncodedNumber[leaf_cur_index];
        client.share_batch_decrypt(label_vector, decrypted_labels, leaf_cur_index);
        for (int i = 0; i < leaf_cur_index; i++) {
            float x;
            decrypted_labels[i].decode(x);
            logger(logger_out, "decoded label vector[%d] = %f\n", i, x);
            new_label_vector[i].set_float(client.m_pk->n[0], x);
            djcs_t_aux_encrypt(client.m_pk, client.m_hr, new_label_vector[i], new_label_vector[i]);
        }
        delete [] decrypted_labels;
    } else {
        std::string s, response_s;
        client.recv_long_messages(client.client_num - 1, s);
        client.decrypt_batch_piece(s, response_s, client.client_num - 1);
    }

    // for each sample
    for (int i = 0; i < size; ++i) {
        // compute binary vector for the current sample
        std::vector<float> sample_values = input_dataset[i];
        std::vector<int> binary_vector = compute_binary_vector(class_id,
            tree_id, sample_values, node_index_2_leaf_index_map);
        EncodedNumber *encoded_binary_vector = new EncodedNumber[binary_vector.size()];
        EncodedNumber *updated_label_vector;// = new EncodedNumber[binary_vector.size()];
        // update in Robin cycle, from the last client to client 0
        if (client.client_id == client.client_num - 1) {
            updated_label_vector = new EncodedNumber[binary_vector.size()];
            for (int j = 0; j < binary_vector.size(); j++) {
                encoded_binary_vector[j].set_integer(client.m_pk->n[0], binary_vector[j]);
                djcs_t_aux_ep_mul(client.m_pk, updated_label_vector[j],
                    new_label_vector[j], encoded_binary_vector[j]);
            }
            // send to the next client
            std::string send_s;
            serialize_batch_sums(updated_label_vector, binary_vector.size(), send_s);
            client.send_long_messages(client.client_id - 1, send_s);
        } else if (client.client_id > 0) {
            std::string recv_s;
            client.recv_long_messages(client.client_id + 1, recv_s);
            int recv_size; // should be same as binary_vector.size()
            deserialize_sums_from_string(updated_label_vector, recv_size, recv_s);
            for (int j = 0; j < binary_vector.size(); j++) {
                encoded_binary_vector[j].set_integer(client.m_pk->n[0], binary_vector[j]);
                djcs_t_aux_ep_mul(client.m_pk, updated_label_vector[j],
                    updated_label_vector[j], encoded_binary_vector[j]);
            }
            std::string resend_s;
            serialize_batch_sums(updated_label_vector, binary_vector.size(), resend_s);
            client.send_long_messages(client.client_id - 1, resend_s);
        } else {
            // the super client update the last, and aggregate before calling share decryption
            std::string final_recv_s;
            client.recv_long_messages(client.client_id + 1, final_recv_s);
            int final_recv_size;
            deserialize_sums_from_string(updated_label_vector, final_recv_size, final_recv_s);
            for (int j = 0; j < binary_vector.size(); j++) {
                encoded_binary_vector[j].set_integer(client.m_pk->n[0], binary_vector[j]);
                djcs_t_aux_ep_mul(client.m_pk, updated_label_vector[j],
                    updated_label_vector[j], encoded_binary_vector[j]);
            }
        }
        // aggregate
        if (client.client_id == SUPER_CLIENT_ID) {
            EncodedNumber *encrypted_aggregation = new EncodedNumber[1];
            encrypted_aggregation[0].set_float(client.m_pk->n[0], 0, FLOAT_PRECISION);
            djcs_t_aux_encrypt(client.m_pk, client.m_hr,
                encrypted_aggregation[0], encrypted_aggregation[0]);
            for (int j = 0; j < binary_vector.size(); j++) {
                djcs_t_aux_ee_add(client.m_pk, encrypted_aggregation[0],
                    encrypted_aggregation[0], updated_label_vector[j]);
            }
            encrypted_predicted_labels[i] = encrypted_aggregation[0];
            delete [] encrypted_aggregation;
        }
        delete [] encoded_binary_vector;
        delete [] updated_label_vector;
    }
    delete [] label_vector;
    delete [] new_label_vector;
}

void GBDT::test_accuracy(Client & client, float & accuracy) {
    logger(logger_out, "Begin test accuracy on testing dataset\n");
    std::vector<float> predicted_label_vector;
    std::vector< std::vector<float> > predicted_forest_labels;
    for (int class_id = 0; class_id < classes_num; class_id++) {
        std::vector<float> t;
        for (int i = 0; i < testing_data.size(); i++) {
            t.push_back(0.0);
        }
        predicted_forest_labels.push_back(t);
    }
    for (int class_id = 0; class_id < classes_num; class_id++) {
        for (int tree_id = 0; tree_id < num_trees; tree_id++) {
            std::vector<float> labels = compute_predicted_labels(client, class_id, tree_id, 1);
            if (tree_id == 0) {
                for (int i = 0; i < testing_data.size(); i++) {
                    predicted_forest_labels[class_id][i] = predicted_forest_labels[class_id][i] + labels[i];
                }
            } else {
                for (int i = 0; i < testing_data.size(); i++) {
                    predicted_forest_labels[class_id][i] = predicted_forest_labels[class_id][i] +
                        GBDT_LEARNING_RATE * labels[i];
                }
            }
        }
    }
    if (gbdt_type == Classification) {
        for (int i = 0; i < testing_data.size(); i++) {
            std::vector<float> prediction_prob_i;
            for (int class_id = 0; class_id < classes_num; class_id++) {
                prediction_prob_i.push_back(predicted_forest_labels[class_id][i]);
            }
            float label = argmax(prediction_prob_i);
            predicted_label_vector.push_back(label);
        }
    } else {
        for (int i = 0; i < testing_data.size(); i++) {
            predicted_label_vector.push_back(predicted_forest_labels[0][i]);
        }
    }
    // compute accuracy by the super client
    if (client.client_id == SUPER_CLIENT_ID) {
        if (gbdt_type == Classification) {
            int correct_num = 0;
            for (int i = 0; i < testing_data.size(); i++) {
                if (rounded_comparison(predicted_label_vector[i], testing_data_labels[i])) {
                    correct_num += 1;
                }
            }
            logger(logger_out, "correct_num = %d, testing_data_size = %d\n",
                correct_num, testing_data_labels.size());
            accuracy = (float) correct_num / (float) testing_data_labels.size();
        } else {
            accuracy = mean_squared_error(predicted_label_vector, testing_data_labels);
        }
    }
    logger(logger_out, "End test accuracy on testing dataset\n");
}

GBDT::~GBDT() {}