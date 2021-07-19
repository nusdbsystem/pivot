#include <iostream>
#include <boost/program_options.hpp>
#include <string>
#include <random>
#include <ios>
#include <fstream>
#include "src/utils/util.h"
#include "src/utils/encoder.h"
#include "src/utils/djcs_t_aux.h"
#include "src/utils/pb_converter.h"
#include "src/client/client.h"
#include "src/models/cart_tree.h"
#include "src/models/feature.h"
#include "src/models/tree_node.h"
#include "src/models/random_forest.h"
#include "src/models/gbdt.h"

#include "tests/test_encoder.h"
#include "tests/test_djcs_t_aux.h"
#include "tests/test_pb_converter.h"

hcs_random *hr;
djcs_t_public_key *pk;
djcs_t_private_key *vk;
mpz_t n, positive_threshold, negative_threshold;
djcs_t_auth_server **au = (djcs_t_auth_server **)malloc(TOTAL_CLIENT_NUM * sizeof(djcs_t_auth_server *));
mpz_t *si = (mpz_t *)malloc(TOTAL_CLIENT_NUM * sizeof(mpz_t));

FILE * logger_out;
bool gbdt_flag = false;

void system_setup() {
    hr = hcs_init_random();
    pk = djcs_t_init_public_key();
    vk = djcs_t_init_private_key();

    djcs_t_generate_key_pair(pk, vk, hr, 1, 1024, TOTAL_CLIENT_NUM, TOTAL_CLIENT_NUM);
    mpz_t *coeff = djcs_t_init_polynomial(vk, hr);

    for (int i = 0; i < TOTAL_CLIENT_NUM; i++) {
        mpz_init(si[i]);
        djcs_t_compute_polynomial(vk, coeff, si[i], i);
        au[i] = djcs_t_init_auth_server();
        djcs_t_set_auth_server(au[i], si[i], i);
    }
    mpz_init(n);
    mpz_init(positive_threshold);
    mpz_init(negative_threshold);
    compute_thresholds(pk, n, positive_threshold, negative_threshold);

    djcs_t_free_polynomial(vk, coeff);
}

void system_free() {
    // free memory
    hcs_free_random(hr);
    djcs_t_free_public_key(pk);
    djcs_t_free_private_key(vk);
    mpz_clear(n);
    mpz_clear(positive_threshold);
    mpz_clear(negative_threshold);
    for (int i = 0; i < TOTAL_CLIENT_NUM; i++) {
        mpz_clear(si[i]);
        djcs_t_free_auth_server(au[i]);
    }
    free(si);
    free(au);
}

float decision_tree(Client & client, int solution_type,
    int optimization_type, int class_num, int tree_type,
    int max_bins, int max_depth, int num_trees) {
    logger(logger_out, "Begin decision tree training\n");
    struct timeval decision_tree_training_1, decision_tree_training_2;
    double decision_tree_training_time = 0;
    gettimeofday(&decision_tree_training_1, NULL);

    int m_global_feature_num = GLOBAL_FEATURE_NUM;
    int m_local_feature_num = client.local_data[0].size();
    int m_internal_node_num = 0;
    int m_type = tree_type;
    int m_classes_num = class_num;
    if (m_type == Regression) m_classes_num = CLASS_NUM_FOR_REGRESSION;
    int m_max_depth = max_depth;
    int m_max_bins = max_bins;
    int m_prune_sample_num = PRUNE_SAMPLE_NUM;
    float m_prune_threshold = PRUNE_VARIANCE_THRESHOLD;
    int m_solution_type = solution_type;
    int m_optimization_type = optimization_type;
    DecisionTree model(m_global_feature_num, m_local_feature_num,
        m_internal_node_num, m_type, m_classes_num,
        m_max_depth, m_max_bins, m_prune_sample_num,
        m_prune_threshold, m_solution_type, m_optimization_type);
    logger(logger_out, "Init decision tree model succeed\n");

    float split = SPLIT_PERCENTAGE;
    if (client.client_id == SUPER_CLIENT_ID) {
        model.init_datasets(client, split);
        //model.test_indicator_vector_correctness();
    } else {
        int *new_indexes = new int[client.sample_num];
        std::string recv_s;
        client.recv_long_messages(SUPER_CLIENT_ID, recv_s);
        deserialize_ids_from_string(new_indexes, recv_s);
        model.init_datasets_with_indexes(client, new_indexes, split);
        delete [] new_indexes;
    }
    logger(logger_out, "Training data size = %d\n", model.training_data.size());

    model.init_features();
    model.init_root_node(client);
    model.build_tree_node(client, 0);
    logger(logger_out, "End decision tree training\n");
    logger(logger_out, "The internal node number is %d\n", model.internal_node_num);

    gettimeofday(&decision_tree_training_2, NULL);
    decision_tree_training_time +=
        (double)((decision_tree_training_2.tv_sec - decision_tree_training_1.tv_sec) * 1000 +
        (double)(decision_tree_training_2.tv_usec - decision_tree_training_1.tv_usec) / 1000);
    logger(logger_out, "*********************************************************************\n");
    logger(logger_out, "******** Decision tree training time: %'.3f ms **********\n", decision_tree_training_time);
    logger(logger_out, "*********************************************************************\n");
    struct timeval decision_tree_prediction_1, decision_tree_prediction_2;
    double decision_tree_prediction_average_time = 0;
    gettimeofday(&decision_tree_prediction_1, NULL);

    float accuracy = 0.0;
    model.test_accuracy(client, accuracy);

    gettimeofday(&decision_tree_prediction_2, NULL);
    decision_tree_prediction_average_time +=
        (double)((decision_tree_prediction_2.tv_sec - decision_tree_prediction_1.tv_sec) * 1000 +
        (double)(decision_tree_prediction_2.tv_usec - decision_tree_prediction_1.tv_usec) / 1000);
    decision_tree_prediction_average_time = decision_tree_prediction_average_time / (double) model.testing_data.size();
    logger(logger_out, "*********************************************************************\n");
    logger(logger_out, "********* Average decision tree prediction time: %'.3f ms ************\n",
        decision_tree_prediction_average_time);
    logger(logger_out, "*********************************************************************\n");

    if (client.client_id == SUPER_CLIENT_ID) {
        logger(logger_out, "Accuracy = %f\n", accuracy);
//        std::string result_log_file = LOGGER_HOME;
//        result_log_file += "result.log";
//        std::ofstream result_log(result_log_file, std::ios_base::app | std::ios_base::out);
//        result_log << accuracy << std::endl;
    }
    return accuracy;
}

float random_forest(Client & client, int solution_type,
    int optimization_type, int class_num, int tree_type,
    int max_bins, int max_depth, int num_trees) {
    logger(logger_out, "Begin random forest training\n");
    struct timeval random_forest_training_1, random_forest_training_2;
    double random_forest_training_time = 0;
    gettimeofday(&random_forest_training_1, NULL);

    int m_tree_num = num_trees;
    int m_global_feature_num = GLOBAL_FEATURE_NUM;
    int m_local_feature_num = client.local_data[0].size();
    int m_internal_node_num = 0;
    int m_type = tree_type;
    int m_classes_num = class_num;
    if (m_type == Regression) m_classes_num = CLASS_NUM_FOR_REGRESSION;
    int m_max_depth = max_depth;
    int m_max_bins = max_bins;
    int m_prune_sample_num = PRUNE_SAMPLE_NUM;
    float m_prune_threshold = PRUNE_VARIANCE_THRESHOLD;
    RandomForest model(m_tree_num, m_global_feature_num, m_local_feature_num,
        m_internal_node_num, m_type, m_classes_num,
        m_max_depth, m_max_bins, m_prune_sample_num,
        m_prune_threshold, solution_type, optimization_type);
    // split datasets to training part and testing part
    float split = SPLIT_PERCENTAGE;
    if (client.client_id == SUPER_CLIENT_ID) {
        model.init_datasets(client, split);
    } else {
        int *new_indexes = new int[client.sample_num];
        std::string recv_s;
        client.recv_long_messages(SUPER_CLIENT_ID, recv_s);
        deserialize_ids_from_string(new_indexes, recv_s);
        model.init_datasets_with_indexes(client, new_indexes, split);
        delete [] new_indexes;
    }
    float sample_rate = RF_SAMPLE_RATE;
    model.build_forest(client, sample_rate);

    gettimeofday(&random_forest_training_2, NULL);
    random_forest_training_time +=
        (double)((random_forest_training_2.tv_sec - random_forest_training_1.tv_sec) * 1000 +
        (double)(random_forest_training_2.tv_usec - random_forest_training_1.tv_usec) / 1000);
    logger(logger_out, "*********************************************************************\n");
    logger(logger_out, "******** Random forest training time: %'.3f ms **********\n", random_forest_training_time);
    logger(logger_out, "*********************************************************************\n");

    struct timeval random_forest_prediction_1, random_forest_prediction_2;
    double random_forest_prediction_average_time = 0;
    gettimeofday(&random_forest_prediction_1, NULL);

    float accuracy = 0.0;
    model.test_accuracy(client, accuracy);

    gettimeofday(&random_forest_prediction_2, NULL);
    random_forest_prediction_average_time +=
        (double)((random_forest_prediction_2.tv_sec - random_forest_prediction_1.tv_sec) * 1000 +
        (double)(random_forest_prediction_2.tv_usec - random_forest_prediction_1.tv_usec) / 1000);
    random_forest_prediction_average_time = random_forest_prediction_average_time / (double) model.testing_data.size();
    logger(logger_out, "*********************************************************************\n");
    logger(logger_out, "********* Average random forest prediction time: %'.3f ms ************\n",
        random_forest_prediction_average_time);
    logger(logger_out, "*********************************************************************\n");

    if (client.client_id == SUPER_CLIENT_ID) {
        logger(logger_out, "Accuracy = %f\n", accuracy);
//        std::string result_log_file = LOGGER_HOME;
//        result_log_file += "result.log";
//        std::ofstream result_log(result_log_file, std::ios_base::app | std::ios_base::out);
//        result_log << accuracy << std::endl;
    }
    return accuracy;
}

float gbdt(Client & client, int solution_type,
    int optimization_type, int class_num, int tree_type,
    int max_bins, int max_depth, int num_trees) {
    logger(logger_out, "Begin GBDT training\n");
    struct timeval gbdt_training_1, gbdt_training_2;
    double gbdt_training_time = 0;
    gettimeofday(&gbdt_training_1, NULL);

    int m_tree_num = num_trees;
    int m_global_feature_num = GLOBAL_FEATURE_NUM;
    int m_local_feature_num = client.local_data[0].size();
    int m_internal_node_num = 0;
    int m_type = tree_type;
    int m_classes_num = class_num;
    if (m_type == Regression) m_classes_num = CLASS_NUM_FOR_REGRESSION;
    int m_max_depth = max_depth;
    int m_max_bins = max_bins;
    int m_prune_sample_num = PRUNE_SAMPLE_NUM;
    float m_prune_threshold = PRUNE_VARIANCE_THRESHOLD;
    GBDT model(m_tree_num, m_global_feature_num, m_local_feature_num,
        m_internal_node_num, m_type, m_classes_num,
        m_max_depth, m_max_bins, m_prune_sample_num,
        m_prune_threshold, solution_type, optimization_type);
    logger(logger_out, "Correct init gbdt\n");

    float split = SPLIT_PERCENTAGE;
    if (client.client_id == SUPER_CLIENT_ID) {
        model.init_datasets(client, split);
        //model.test_indicator_vector_correctness();
    } else {
        int *new_indexes = new int[client.sample_num];
        std::string recv_s;
        client.recv_long_messages(SUPER_CLIENT_ID, recv_s);
        deserialize_ids_from_string(new_indexes, recv_s);
        model.init_datasets_with_indexes(client, new_indexes, split);
        delete [] new_indexes;
    }
    model.build_gbdt_with_spdz(client);

    gettimeofday(&gbdt_training_2, NULL);
    gbdt_training_time += (double)((gbdt_training_2.tv_sec - gbdt_training_1.tv_sec) * 1000 +
        (double)(gbdt_training_2.tv_usec - gbdt_training_1.tv_usec) / 1000);
    logger(logger_out, "*********************************************************************\n");
    logger(logger_out, "******** GBDT training time: %'.3f ms **********\n", gbdt_training_time);
    logger(logger_out, "*********************************************************************\n");

    struct timeval gbdt_prediction_1, gbdt_prediction_2;
    double gbdt_prediction_average_time = 0;
    gettimeofday(&gbdt_prediction_1, NULL);

    float accuracy = 0.0;
    model.test_accuracy(client, accuracy);

    gettimeofday(&gbdt_prediction_2, NULL);
    gbdt_prediction_average_time +=
        (double)((gbdt_prediction_2.tv_sec - gbdt_prediction_1.tv_sec) * 1000 +
        (double)(gbdt_prediction_2.tv_usec - gbdt_prediction_1.tv_usec) / 1000);
    gbdt_prediction_average_time = gbdt_prediction_average_time / (double) model.testing_data.size();
    logger(logger_out, "*********************************************************************\n");
    logger(logger_out, "********* Average GBDT prediction time: %'.3f ms ************\n", gbdt_prediction_average_time);
    logger(logger_out, "*********************************************************************\n");

    //model.test_accuracy_with_spdz(client, accuracy);
    if (client.client_id == SUPER_CLIENT_ID) {
        logger(logger_out, "Accuracy = %f\n", accuracy);
//        std::string result_log_file = LOGGER_HOME;
//        result_log_file += "result.log";
//        std::ofstream result_log(result_log_file, std::ios_base::app | std::ios_base::out);
//        result_log << accuracy << std::endl;
    }
    return accuracy;
}

int main(int argc, char *argv[]) {
    int client_id, client_num, class_num, algorithm_type, tree_type;
    int solution_type, optimization_type, max_bins, max_depth, num_trees;
    std::string network_file, data_file, logger_file_name;

    try {
        namespace po = boost::program_options;
        po::options_description description("Usage:");
        description.add_options()
            ("help,h", "display this help message")
            ("version,v", "display the version number")
            ("client-id", po::value<int>(&client_id), "current client id")
            ("client-num", po::value<int>(&client_num), "total client num")
            ("class-num", po::value<int>(&class_num), "num of classes for the task")
            ("algorithm-type", po::value<int>(&algorithm_type), "desired algorithm, decision tree, random forest, or gbdt")
            ("tree-type", po::value<int>(&tree_type), "classification or regression")
            ("solution-type", po::value<int>(&solution_type), "basic protocol or enhanced protocol")
            ("optimization-type", po::value<int>(&optimization_type), "optimization used for this protocol")
            ("network-file", po::value<std::string>(&network_file), "network file used")
            ("data-file", po::value<std::string>(&data_file), "dataset used for the task")
            ("logger-file", po::value<std::string>(&logger_file_name), "logger file header")
            ("max-bins", po::value<int>(&max_bins), "maximum bins for splitting each feature")
            ("max-depth", po::value<int>(&max_depth), "maximum tree depth")
            ("num-trees", po::value<int>(&num_trees), "num of trees in ensemble models");

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(description).run(), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << "Usage: options_description [options]\n";
            std::cout << description;
            return 0;
        }
        std::cout << "parse parameters correct" << std::endl;

        std::cout << "client-id: " << vm["client-id"].as<int>() << std::endl;
        std::cout << "client-num: " << vm["client-num"].as<int>() << std::endl;
        std::cout << "class-num: " << vm["class-num"].as<int>() << std::endl;
        std::cout << "algorithm-type: " << vm["algorithm-type"].as<int>() << std::endl;
        std::cout << "tree-type: " << vm["tree-type"].as<int>() << std::endl;
        std::cout << "solution-type: " << vm["solution-type"].as<int>() << std::endl;
        std::cout << "optimization-type: " << vm["optimization-type"].as<int>() << std::endl;
        std::cout << "network-file: " << vm["network-file"].as< std::string >() << std::endl;
        std::cout << "data-file: " << vm["data-file"].as< std::string >() << std::endl;
        std::cout << "logger-file: " << vm["logger-file"].as< std::string >() << std::endl;
        std::cout << "max-bins: " << vm["max-bins"].as<int>() << std::endl;
        std::cout << "max-depth: " << vm["max-depth"].as<int>() << std::endl;
        std::cout << "num-trees: " << vm["num-trees"].as<int>() << std::endl;
    }
    catch(std::exception& e)
    {
        cout << e.what() << "\n";
        return 1;
    }

    //test_pb();

    if (client_id == SUPER_CLIENT_ID) {
        system_setup();
    }
    // create logger file
    std::string alg_name;
    switch (algorithm_type) {
        case RandomForestAlg:
            alg_name = "RF";
            break;
        case GBDTAlg:
            alg_name = "GBDT";
            break;
        default:
            alg_name = "DT";
            break;
    }
    logger_file_name += "_";
    logger_file_name += alg_name;
    logger_file_name += "_";
    logger_file_name += get_timestamp_str();
    logger_file_name += "_client";
    logger_file_name += to_string(client_id);
    logger_file_name += ".txt";
    logger_out = fopen(logger_file_name.c_str(), "wb");

    bool has_label = (client_id == SUPER_CLIENT_ID);

    Client client(client_id, client_num, has_label, network_file, data_file);
    // set up keys
    if (client.client_id == SUPER_CLIENT_ID) {
        client.set_keys(pk, hr, si[client.client_id], client.client_id);
        // send keys
        for (int i = 0; i < client.client_num; i++) {
            if (i != client.client_id) {
                std::string keys_i;
                client.serialize_send_keys(keys_i, pk, si[i], i);
                client.send_long_messages(i, keys_i);
            }
        }
    } else {
        // receive keys from client 0
        std::string recv_keys;
        client.recv_long_messages(SUPER_CLIENT_ID, recv_keys);
        client.recv_set_keys(recv_keys);
        mpz_init(n);
        mpz_init(positive_threshold);
        mpz_init(negative_threshold);
        compute_thresholds(client.m_pk, n, positive_threshold, negative_threshold);
    }

    switch(algorithm_type) {
        case RandomForestAlg:
            random_forest(client, solution_type, optimization_type,
                class_num, tree_type, max_bins, max_depth, num_trees);
            break;
        case GBDTAlg:
            gbdt_flag = true;
            gbdt(client, solution_type, optimization_type,
                class_num, tree_type, max_bins, max_depth, num_trees);
            break;
        default:
            decision_tree(client, solution_type, optimization_type,
                class_num, tree_type, max_bins, max_depth, num_trees);
            break;
    }

    if (client_id == SUPER_CLIENT_ID) {
        system_free();
    } else {
        mpz_clear(n);
        mpz_clear(positive_threshold);
        mpz_clear(negative_threshold);
    }
    logger(logger_out, "The End\n");
    return 0;
}