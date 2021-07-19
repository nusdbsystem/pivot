//
// Created by wuyuncheng on 20/11/19.
//

#include "spdz_util.h"
#include "math.h"
#include "../util.h"
#include "../../include/common.h"

extern FILE * logger_out;

void send_private_batch_shares(std::vector<float> shares, std::vector<int>& sockets, int n_parties) {
    int number_inputs = shares.size();
    std::vector<int64_t> long_shares(number_inputs);
    // step 1: convert to int or long according to the fixed precision
    for (int i = 0; i < number_inputs; ++i) {
        long_shares[i] = static_cast<int64_t>(round(shares[i] * pow(2, SPDZ_FIXED_PRECISION)));
    }
    // step 2: convert to the gfp value and call send_private_inputs
    // Map inputs into gfp
    vector<gfp> input_values_gfp(number_inputs);
    for (int i = 0; i < number_inputs; i++) {
        input_values_gfp[i].assign(long_shares[i]);
    }
    // Run the computation
    send_private_inputs(input_values_gfp, sockets, n_parties);
}

void send_private_batch_shares_packing(std::vector<float> shares, std::vector<int>& sockets, int n_parties) {
    int number_inputs = shares.size();
    // store the base
    int64_t base = pow(2, SPDZ_FIXED_PRECISION);
    gfp helper;
    helper.assign(base);
    // init the values
    vector<gfp> input_values_gfp(number_inputs);
    for (int i = 0; i < number_inputs; i++) {
        int64_t aa = round(shares[i]);
        input_values_gfp[i].assign(aa);
        input_values_gfp[i].mul(helper);
    }
    // run the computation
    send_private_inputs(input_values_gfp, sockets, n_parties);
}

void send_public_parameters(int type, int global_split_num, int classes_num, int used_classes_num, std::vector<int>& sockets, int n_parties) {
    octetStream os;
    vector<gfp> parameters(4);
    parameters[0].assign(type);
    parameters[1].assign(global_split_num);
    parameters[2].assign(classes_num);
    parameters[3].assign(used_classes_num);

    parameters[0].pack(os);
    parameters[1].pack(os);
    parameters[2].pack(os);
    parameters[3].pack(os);
    for (int i = 0; i < n_parties; i++) {
        os.Send(sockets[i]);
    }
}

void send_public_values(std::vector<int> values, std::vector<int>& sockets, int n_parties) {
    octetStream os;
    int size = values.size();
    vector<gfp> parameters(size);
    for (int i = 0; i < size; i++) {
        parameters[i].assign(values[i]);
        parameters[i].pack(os);
    }
    for (int i = 0; i < n_parties; i++) {
        os.Send(sockets[i]);
    }
}

std::vector<int> setup_sockets(int n_parties, int my_client_id, std::vector<std::string> host_names, int port_base) {
    // Setup connections from this client to each party socket
    std::vector<int> sockets(n_parties);
    for (int i = 0; i < n_parties; i++)
    {
        set_up_client_socket(sockets[i], host_names[i].c_str(), port_base + i);
        send(sockets[i], (octet*) &my_client_id, sizeof(int));
        //cout << "set up for " << i << "-th party succeed" << ", sockets = " << sockets[i] << ", port_num = " << port_base + i << endl;
    }
    //cout << "Finish setup socket connections to SPDZ engines." << endl;
    return sockets;
}

void send_private_inputs(const std::vector<gfp>& values, std::vector<int>& sockets, int n_parties)
{
    int num_inputs = values.size();
    octetStream os;
    std::vector< std::vector<gfp> > triples(num_inputs, vector<gfp>(3));
    std::vector<gfp> triple_shares(3);
    // Receive num_inputs triples from SPDZ
    for (int j = 0; j < n_parties; j++)
    {
        os.reset_write_head();
        os.Receive(sockets[j]);

        for (int j = 0; j < num_inputs; j++)
        {
            for (int k = 0; k < 3; k++)
            {
                triple_shares[k].unpack(os);
                triples[j][k] += triple_shares[k];
            }
        }
    }
    // Check triple relations (is a party cheating?)
    for (int i = 0; i < num_inputs; i++)
    {
        if (triples[i][0] * triples[i][1] != triples[i][2])
        {
            cerr << "Incorrect triple at " << i << ", aborting\n";
            exit(1);
        }
    }
    // Send inputs + triple[0], so SPDZ can compute shares of each value
    os.reset_write_head();
    for (int i = 0; i < num_inputs; i++)
    {
        gfp y = values[i] + triples[i][0];
        y.pack(os);
    }
    for (int j = 0; j < n_parties; j++)
        os.Send(sockets[j]);
}

void initialise_fields(const string& dir_prefix)
{
    int lg2;
    bigint p;
    string filename = DEFAULT_PARAM_DATA_FILE;
    // string filename = "/home/sunxutao/projects/VFL-SPDZ/Player-Data/3-128-128/Params-Data";
    //string filename = dir_prefix + "Params-Data";
    //logger(logger_out, "loading params for SPDZ from %s\n", filename.c_str());
    ifstream inpf(filename.c_str());
    if (inpf.fail()) { throw file_error(filename.c_str()); }
    inpf >> p;
    inpf >> lg2;
    inpf.close();
    gfp::init_field(p);
    //gf2n::init_field(lg2);
}

std::vector<float> receive_result(std::vector<int>& sockets, int n_parties, int size)
{
    std::vector<gfp> output_values(size);
    octetStream os;
    for (int i = 0; i < n_parties; i++)
    {
        os.reset_write_head();
        os.Receive(sockets[i]);
        for (int j = 0; j < size; j++)
        {
            gfp value;
            value.unpack(os);
            output_values[j] += value;
        }
    }
    std::vector<float> res_shares(size);
    for (int i = 0; i < size; i++) {
        gfp val = output_values[i];
        bigint aa;
        to_signed_bigint(aa, val);
        int64_t t = aa.get_si();
        //cout<< "i = " << i << ", t = " << t <<endl;
        res_shares[i] = static_cast<float>(t * pow(2, -SPDZ_FIXED_PRECISION));
    }
    return res_shares;
}

std::vector<float> receive_result_dt(std::vector<int>& sockets, int n_parties, int size, int & best_split_index) {
    logger(logger_out, "Receive result from the SPDZ engine\n");
    std::vector<gfp> output_values(size);
    octetStream os;
    for (int i = 0; i < n_parties; i++)
    {
        os.reset_write_head();
        os.Receive(sockets[i]);
        for (int j = 0; j < size; j++)
        {
            gfp value;
            value.unpack(os);
            output_values[j] += value;
        }
    }
    std::vector<float> res_shares(size - 1);
    for (int i = 0; i < size - 1; i++) {
        gfp val = output_values[i];
        bigint aa;
        to_signed_bigint(aa, val);
        int64_t t = aa.get_si();
        //cout<< "i = " << i << ", t = " << t <<endl;
        res_shares[i] = static_cast<float>(t * pow(2, -SPDZ_FIXED_PRECISION));
    }
    gfp index = output_values[size - 1];
    bigint index_aa;
    to_signed_bigint(index_aa, index);
    best_split_index = index_aa.get_si();
    return res_shares;
}

std::vector<float> receive_mode(std::vector<int>& sockets, int n_parties, int size) {
    //logger(logger_out, "Receive mode from the SPDZ engine\n");
    std::vector<gfp> output_values(size);
    octetStream os;
    for (int i = 0; i < n_parties; i++)
    {
        os.reset_write_head();
        os.Receive(sockets[i]);
        for (int j = 0; j < size; j++)
        {
            gfp value;
            value.unpack(os);
            output_values[j] += value;
        }
    }
    std::vector<float> modes(size);
    for (int i = 0; i < size; i++) {
        gfp val = output_values[i];
        bigint aa;
        to_signed_bigint(aa, val);
        int64_t t = aa.get_si();
        //cout<< "i = " << i << ", t = " << t <<endl;
        modes[i] = static_cast<float>(t * pow(2, -SPDZ_FIXED_PRECISION));
    }
    return modes;
}