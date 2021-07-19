//
// Created by wuyuncheng on 10/10/19.
//

#include "client.h"
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <stack>
#include <infra/ConfigFile.hpp>
#include "libhcs.h"
#include "gmp.h"
#include <comm/Comm.hpp>
#include "../utils/util.h"
#include "../utils/pb_converter.h"
#include "../include/protobuf/keys.pb.h"
#include "../utils/djcs_t_aux.h"
#include "omp.h"
extern FILE * logger_out;

Client::Client() {}

Client::Client(int param_client_id, int param_client_num, int param_has_label,
                std::string param_network_config_file, std::string param_local_data_file)
{
    // copy params
    client_id = param_client_id;
    client_num = param_client_num;
    has_label = param_has_label == 1;
    // init threshold DamgardJurik keys
    m_pk = djcs_t_init_public_key();
    m_au = djcs_t_init_auth_server();
    m_hr = hcs_init_random();
    // read local dataset
    std::ifstream data_infile(param_local_data_file);
    if (!data_infile) {
        logger(logger_out, "open %s error\n", param_local_data_file.c_str());
        exit(EXIT_FAILURE);
    }
    std::string line;
    while (std::getline(data_infile, line)) {
        std::vector<float> items;
        std::istringstream ss(line);
        std::string item;
        // split line with delimiter, default ','
        while(getline(ss, item,','))
        {
            items.push_back(::atof(item.c_str()));
        }
        local_data.push_back(items);
    }
    sample_num = local_data.size();
    feature_num = local_data[0].size();
    if (has_label) {
        // slice the last item as label
        for (int i = 0; i < sample_num; i++) {
            labels.push_back(local_data[i][feature_num-1]);
            local_data[i].pop_back();
        }
        feature_num = feature_num - 1;
    }

    // read network config file
    ConfigFile config_file(param_network_config_file);
    string portString, ipString;
    vector<int> ports(client_num);
    vector<string> ips(client_num);
    for (int i = 0; i < client_num; i++) {
        portString = "party_" + to_string(i) + "_port";
        ipString = "party_" + to_string(i) + "_ip";
        //get parties ips and ports
        ports[i] = stoi(config_file.Value("",portString));
        ips[i] = config_file.Value("",ipString);
    }
    host_names = ips;

    // establish connections
    SocketPartyData me, other;
    for (int  i = 0;  i < client_num; ++ i) {
        if (i < client_id) {
            // this party will be the receiver in the protocol
            me = SocketPartyData(boost_ip::address::from_string(ips[client_id]), ports[client_id] + i);
            logger(logger_out, "my port = %d\n", ports[client_id] + i);
            other = SocketPartyData(boost_ip::address::from_string(ips[i]), ports[i] + client_id - 1);
            logger(logger_out, "other port = %d\n", ports[i] + client_id - 1);
            shared_ptr<CommParty> channel = make_shared<CommPartyTCPSynced>(io_service, me, other);
            // connect to the other party
            channel->join(500,5000);
            logger(logger_out, "channel established\n");
            // add channel to the other client
            channels.push_back(std::move(channel));
        } else if (i > client_id) {
            // this party will be the sender in the protocol
            me = SocketPartyData(boost_ip::address::from_string(ips[client_id]), ports[client_id] + i - 1);
            logger(logger_out, "my port = %d\n", ports[client_id] + i - 1);
            other = SocketPartyData(boost_ip::address::from_string(ips[i]), ports[i] + client_id);
            logger(logger_out, "other port = %d\n", ports[i] + client_id);
            shared_ptr<CommParty> channel = make_shared<CommPartyTCPSynced>(io_service, me, other);
            // connect to the other party
            channel->join(500,5000);
            logger(logger_out, "channel established\n");
            // add channel to the other client
            channels.push_back(std::move(channel));
        } else {
            // self communication
            shared_ptr<CommParty> channel = NULL;
            channels.push_back(std::move(channel));
        }
    }
    data_infile.close();
}

Client::Client(const Client &client) {
    feature_num = client.feature_num;
    sample_num = client.sample_num;
    client_id = client.client_id;
    client_num = client.client_num;
    has_label = client.has_label;
    local_data = client.local_data;
    labels = client.labels;
    channels = client.channels;
    m_pk = djcs_t_init_public_key();
    m_au = djcs_t_init_auth_server();
    m_hr = hcs_init_random();
    // copy values of pk
    m_pk->s = client.m_pk->s;
    m_pk->l = client.m_pk->l;
    m_pk->w = client.m_pk->w;
    mpz_set(m_pk->g, client.m_pk->g);
    mpz_set(m_pk->delta, client.m_pk->delta);
    m_pk->n = (mpz_t *) malloc(sizeof(mpz_t) * (m_pk->s+1));
    for (int i = 0; i < m_pk->s + 1; i++) {
        mpz_init_set(m_pk->n[i], client.m_pk->n[i]);
    }
    // copy values of hr
    // gmp_randinit_set(m_hr->rstate, client.m_hr->rstate);
    // copay values of au
    m_au->i = client.m_au->i;
    mpz_init_set(m_au->si, client.m_au->si);
}

bool Client::generate_djcs_t_keys(int epsilon_s, int key_size, int client_num, int required_client_num)
{
    // Initialize data structures
    djcs_t_public_key *param_pk = djcs_t_init_public_key();
    djcs_t_private_key *param_vk = djcs_t_init_private_key();
    hcs_random *param_hr = hcs_init_random();
    // Generate a key pair with modulus of size key_size bits
    djcs_t_generate_key_pair(param_pk, param_vk, param_hr, epsilon_s,
            key_size, required_client_num, client_num);
    mpz_t *coeff = djcs_t_init_polynomial(param_vk, param_hr);
    djcs_t_auth_server **param_au =
            (djcs_t_auth_server **)malloc(client_num * sizeof(djcs_t_auth_server *));
    mpz_t *si = (mpz_t *)malloc(required_client_num * sizeof(mpz_t));
    for (int i = 0; i < client_num; i++) {
        mpz_init(si[i]);
        djcs_t_compute_polynomial(param_vk, coeff, si[i], i);
        // TODO: send to corresponding client via network (param_pk, param_hr, si, i)
        //param_au[i] = pcs_t_init_auth_server();
        //pcs_t_set_auth_server(param_au[i], si[i], i);
    }
}

void Client::set_keys(djcs_t_public_key *param_pk, hcs_random *param_hr, mpz_t si, unsigned long i)
{
    // set the keys when receiving from the trusted third party
    // copy values of pk
    m_pk->s = param_pk->s;
    m_pk->l = param_pk->l;
    m_pk->w = param_pk->w;
    mpz_set(m_pk->g, param_pk->g);
    mpz_set(m_pk->delta, param_pk->delta);
    m_pk->n = (mpz_t *) malloc(sizeof(mpz_t) * (m_pk->s+1));
    for (int i = 0; i < m_pk->s + 1; i++) {
        mpz_init_set(m_pk->n[i], param_pk->n[i]);
    }
    // copy values of hr
    // gmp_randinit_set(m_hr->rstate, param_hr->rstate);
    //m_pk = param_pk;
    //m_hr = param_hr;
    djcs_t_set_auth_server(m_au, si, i);
}

void Client::recv_set_keys(std::string recv_keys) {
    com::collaborative::ml::PB_Keys deserialized_pb_keys;
    if (!deserialized_pb_keys.ParseFromString(recv_keys)) {
        logger(logger_out, "Failed to parse PB_Keys from string\n");
        return;
    }
    // set public key
    m_pk->s = deserialized_pb_keys.public_key_s();
    m_pk->l = deserialized_pb_keys.public_key_l();
    m_pk->w = deserialized_pb_keys.public_key_w();
    mpz_t g, delta;
    mpz_init(g);
    mpz_init(delta);
    mpz_set_str(g, deserialized_pb_keys.public_key_g().c_str(), 10);
    mpz_set_str(delta, deserialized_pb_keys.public_key_delta().c_str(), 10);
    mpz_set(m_pk->g, g);
    mpz_set(m_pk->delta, delta);
    m_pk->n = (mpz_t *) malloc(sizeof(mpz_t) * (m_pk->s+1));
    for (int i = 0; i < deserialized_pb_keys.public_key_n_size(); i++) {
        mpz_t ni;
        mpz_init(ni);
        mpz_set_str(ni, deserialized_pb_keys.public_key_n(i).n().c_str(), 10);
        mpz_init_set(m_pk->n[i], ni);
        mpz_clear(ni);
    }
    // set auth server
    mpz_t si;
    mpz_init(si);
    mpz_set_str(si, deserialized_pb_keys.auth_server_si().c_str(), 10);
    djcs_t_set_auth_server(m_au, si, deserialized_pb_keys.auth_server_i());
//    gmp_printf("m_pk->g = %Zd\n", m_pk->g);
//    gmp_printf("m_pk->delta = %Zd\n", m_pk->delta);
//    gmp_printf("m_au->si = %Zd\n", m_au->si);
    mpz_clear(g);
    mpz_clear(delta);
    mpz_clear(si);
}


void Client::serialize_send_keys(std::string &send_keys, djcs_t_public_key *pk, mpz_t si, int i) {
    com::collaborative::ml::PB_Keys pb_keys;
    // serialize public key
    pb_keys.set_public_key_s(pk->s);
    pb_keys.set_public_key_w(pk->w);
    pb_keys.set_public_key_l(pk->l);
    char * g_str_c, * delta_str_c;
    g_str_c = mpz_get_str(NULL, 10, pk->g);
    delta_str_c = mpz_get_str(NULL, 10, pk->delta);
    std::string g_str(g_str_c), delta_str(delta_str_c);
    pb_keys.set_public_key_g(g_str);
    pb_keys.set_public_key_delta(delta_str);
    if (pk->n != NULL) {
        for (int i = 0; i < pk->s + 1; i++) {
            com::collaborative::ml::PB_PublicN *pn = pb_keys.add_public_key_n();
            char * ni_str_c;
            ni_str_c = mpz_get_str(NULL, 10, pk->n[i]);
            std::string ni_str(ni_str_c);
            pn->set_n(ni_str);
            free(ni_str_c);
            //pb_keys.set_public_key_n(i, ni_str);
        }
    }
    // serialize auth server
    pb_keys.set_auth_server_i(i);
    char * si_str_c;
    si_str_c = mpz_get_str(NULL, 10, si);
    std::string si_str(si_str_c);
    pb_keys.set_auth_server_si(si_str);

    pb_keys.SerializeToString(&send_keys);

    free(g_str_c);
    free(delta_str_c);
    free(si_str_c);
}


void Client::share_batch_decrypt(EncodedNumber *ciphers,
    EncodedNumber *& decrypted_res, int size, int parallel) {
    mpz_t **dec = (mpz_t **) malloc (size * sizeof(mpz_t *));
    for (int i = 0; i < size; i++) {
        dec[i] = (mpz_t *) malloc (client_num * sizeof(mpz_t));
        // copy ciphers attributes to decrypted_res
        mpz_set(decrypted_res[i].n, ciphers[i].n);
        decrypted_res[i].exponent = ciphers[i].exponent;
        decrypted_res[i].type = Plaintext;
    }
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < client_num; j++) {
            mpz_init(dec[i][j]);
        }
    }
    // decrypt by its own
    omp_set_num_threads(NUM_OMP_THREADS);
#pragma omp parallel for if (parallel == 1)
    for (int i = 0; i < size; i++) {
        djcs_t_share_decrypt(m_pk, m_au, dec[i][client_id], ciphers[i].value);
    }

    // serialize ciphers and send to the other clients
    std::string send_ciphers;
    serialize_batch_sums(ciphers, size, send_ciphers);
    for (int j = 0; j < client_num; j++) {
        if (j != client_id) {
            std::string recv_message_i;
            EncodedNumber *recv_share_i; // = new EncodedNumber[size];
            send_long_messages(j, send_ciphers);

            // receive the response messages and deserialize to decrypted shares
            recv_long_messages(j, recv_message_i);

            int x;
            deserialize_sums_from_string(recv_share_i, x, recv_message_i);
            for (int i = 0; i < size; i++) {
                mpz_set(dec[i][j], recv_share_i[i].value);
            }
            delete [] recv_share_i;
        }
    }

    // combine all the shares and get the final plaintexts
    for (int i = 0; i < size; i++) {
        mpz_t t;
        mpz_init(t);
        djcs_t_share_combine(m_pk, t, dec[i]);
        mpz_set(decrypted_res[i].value, t);
        mpz_clear(t);
    }

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < client_num; j++) {
            mpz_clear(dec[i][j]);
        }
    }

    for (int i = 0; i < size; i++) {
        free(dec[i]);
    }
    free(dec);
}

void Client::decrypt_batch_piece(std::string s,
    std::string & response_s, int src_client_id, int parallel) {
    // deserialization
    EncodedNumber *ciphers;
    int size;
    deserialize_sums_from_string(ciphers, size, s);
    // share decrypt
    omp_set_num_threads(NUM_OMP_THREADS);
#pragma omp parallel for if (parallel == 1)
    for (int i = 0; i < size; i++) {
//        mpz_t t;
//        mpz_init(t);
//        djcs_t_share_decrypt(m_pk, m_au, t, ciphers[i].value);
//        ciphers[i].type = Plaintext;
//        mpz_set(ciphers[i].value, t);
//        mpz_clear(t);
        djcs_t_share_decrypt(m_pk, m_au, ciphers[i].value, ciphers[i].value);
        ciphers[i].type = Plaintext;
    }
    // serialization and return
    serialize_batch_sums(ciphers, size, response_s);
    send_long_messages(src_client_id, response_s);
    delete [] ciphers;
}

void Client::ciphers_conversion_to_shares(EncodedNumber *src_ciphers,
    std::vector<float> & shares, int size, int precision) {
    if (client_id == SUPER_CLIENT_ID) {
        EncodedNumber * aggregated_encrypted_shares = new EncodedNumber[size];
        for (int i = 0; i < size; i++) {
            aggregated_encrypted_shares[i] = src_ciphers[i];
        }
        // receive and aggregate encrypted shares
        for (int c = 0; c < client_num; c++) {
            if (c != client_id) {
                EncodedNumber * recv_encrypted_shares;// = new EncodedNumber[size];
                int recv_size = 0;
                std::string recv_str_encrypted_shares;
                recv_long_messages(c, recv_str_encrypted_shares);
                deserialize_sums_from_string(recv_encrypted_shares, recv_size, recv_str_encrypted_shares);
                if (recv_size != size) {
                    logger(logger_out, "Ciphers conversion to shares: recv_size not equal to real size\n");
                }
                for (int i = 0; i < size; i++) {
                    djcs_t_aux_ee_add(m_pk, aggregated_encrypted_shares[i],
                        aggregated_encrypted_shares[i], recv_encrypted_shares[i]);
                }
                delete [] recv_encrypted_shares;
            }
        }

        // call share decryption and set shares
        EncodedNumber * decrypted_shares = new EncodedNumber[size];
        share_batch_decrypt(aggregated_encrypted_shares, decrypted_shares, size);
        // decode
        for (int i = 0; i < size; i++) {
            if (precision != 0) {
                float x;
                decrypted_shares[i].decode(x);
                shares.push_back(x);
            } else {
                long v;
                decrypted_shares[i].decode(v);
                shares.push_back((float) v);
            }
        }
        delete [] decrypted_shares;
        delete [] aggregated_encrypted_shares;
    } else {
        // generate secret shares
        EncodedNumber * encrypted_shares = new EncodedNumber[size];
        for (int i = 0; i < size; i++) {
            if (precision != 0) {
                float s = static_cast<float> (rand() % MAXIMUM_RAND_VALUE);
                shares.push_back(0 - s);
                encrypted_shares[i].set_float(m_pk->n[0], s, precision);
                djcs_t_aux_encrypt(m_pk, m_hr, encrypted_shares[i], encrypted_shares[i]);
            } else {
                int s = rand() % MAXIMUM_RAND_VALUE;
                encrypted_shares[i].set_integer(m_pk->n[0], s);
                djcs_t_aux_encrypt(m_pk, m_hr, encrypted_shares[i], encrypted_shares[i]);
                shares.push_back(0 - s);
            }
        }

        // serialize encrypted_shares and send to client 0
        std::string str_encrypted_shares;
        serialize_batch_sums(encrypted_shares, size, str_encrypted_shares);
        send_long_messages(SUPER_CLIENT_ID, str_encrypted_shares);
        // receive share decryption string and decrypt
        std::string recv_str_share_decryption, response_str_share_decryption;
        recv_long_messages(SUPER_CLIENT_ID, recv_str_share_decryption);
        decrypt_batch_piece(recv_str_share_decryption, response_str_share_decryption, 0);

        delete [] encrypted_shares;
    }
}

void Client::cipher_vectors_multiplication(EncodedNumber *cipher_vec1,
    EncodedNumber *cipher_vec2,
    EncodedNumber *&res, int size) {
    /**
     * 1. first convert one cipher vector into secret shares, each client holds a secret share vector
     * 2. then client 0 sends another encrypted cipher vector to the other clients
     * 3. every client locally computes the element-wise homomorphic multiplication between secret share and ciphertext
     * 4. every client sends the intermediate result to client 0, and client 0 aggregates to obtain the final result
     */
    std::vector<float> cipher_vec1_shares;
    EncodedNumber * encoded_cipher_vec1_shares = new EncodedNumber[size];

    // step 1
    ciphers_conversion_to_shares(cipher_vec1, cipher_vec1_shares, size);
    for (int i = 0; i < size; i++) {
        encoded_cipher_vec1_shares[i].set_float(m_pk->n[0], cipher_vec1_shares[i]);
    }
    // step 2
    EncodedNumber * copy_cipher_vec2;// = new EncodedNumber[size];
    if (client_id == SUPER_CLIENT_ID) {
        std::string cipher_vec2_str;
        serialize_batch_sums(cipher_vec2, size, cipher_vec2_str);
        for (int i = 0; i < client_num; i++) {
            if (i != client_id) {
                send_long_messages(i, cipher_vec2_str);
            }
        }
        copy_cipher_vec2 = new EncodedNumber[size];
        for (int i = 0; i < size; i++) {
            copy_cipher_vec2[i] = cipher_vec2[i];
        }
    } else {
        int recv_size = 0;
        std::string recv_cipher_vec2_str;
        recv_long_messages(SUPER_CLIENT_ID, recv_cipher_vec2_str);
        deserialize_sums_from_string(copy_cipher_vec2, recv_size, recv_cipher_vec2_str);
    }
    // step 3
    EncodedNumber * local_multiplication_res = new EncodedNumber[size];
    for (int i = 0; i < size; i++) {
        djcs_t_aux_ep_mul(m_pk, local_multiplication_res[i], copy_cipher_vec2[i], encoded_cipher_vec1_shares[i]);
    }
    // step 4
    if (client_id == SUPER_CLIENT_ID) {
        for (int i = 0; i < size; i++) {
            res[i] = local_multiplication_res[i];
        }
        for (int i = 0; i < client_num; i++) {
            if (i != client_id) {
                std::string recv_local_res_str;
                int recv_size = 0;
                EncodedNumber * recv_local_multiplication_res;// = new EncodedNumber[size];
                recv_long_messages(i, recv_local_res_str);
                deserialize_sums_from_string(recv_local_multiplication_res, recv_size, recv_local_res_str);
                // aggregation
                for (int x = 0; x < size; x++) {
                    djcs_t_aux_ee_add(m_pk, res[x], res[x], recv_local_multiplication_res[x]);
                }
                delete [] recv_local_multiplication_res;
            }
        }
    } else {
        std::string local_res_str;
        serialize_batch_sums(local_multiplication_res, size, local_res_str);
        send_long_messages(SUPER_CLIENT_ID, local_res_str);
    }
    delete [] encoded_cipher_vec1_shares;
    delete [] copy_cipher_vec2;
    delete [] local_multiplication_res;
}

void Client::write_random_shares(std::vector<float> shares, std::string path) {
    ofstream write_file;
    std::string file_name = path + "input/player" + std::to_string(client_id) + ".txt";
    write_file.open(file_name);
    if (!write_file.is_open()) {
        logger(logger_out, "open file %s failed\n", file_name.c_str());
        return;
    }
    std::string str;
    for (int i = 0; i < shares.size(); i++) {
        std::ostringstream ss;
        ss << shares[i];
        string str(ss.str());
        if (i != shares.size() - 1) {
            write_file << str << "\n";
        } else {
            write_file << str;
        }
    }
    write_file.close();
}

std::vector<float> Client::read_random_shares(int size, std::string path) {
    ifstream read_file;
    std::string file_name = path + "output/output" + std::to_string(client_id) + ".txt";
    read_file.open(file_name);
    if (!read_file.is_open()) {
        logger(logger_out, "open file %s failed\n", file_name.c_str());
        exit(1);
    }
    logger(logger_out, "Starting read file\n");
    std::string line;
    std::vector<float> shares;
    while (getline(read_file, line)) {
        if (line != "") {
            float x = ::atof(line.c_str());
            shares.push_back(x);
        }
    }
    if (shares.size() != size) {
        logger(logger_out, "Read share number is not equal to batch size\n");
        exit(1);
    }
    read_file.close();
    return shares;
}

void Client::send_messages(int i, string message) {
    print_send_message(message);
    channels[i]->write((const byte *) message.c_str(), message.size());
}

void Client::send_long_messages(int i, string message) {
    //print_send_message(message);
    channels[i]->writeWithSize(message);
}

void Client::recv_messages(int i, string messages, byte * buffer, int expected_size) {
    channels[i]->read(buffer, expected_size);
    // the size of all strings is 2. Parse the message to get the original strings
    auto s = string(reinterpret_cast<char const*>(buffer), expected_size);
    //print_recv_message(s);
    messages = s;
}

void Client::recv_long_messages(int i, std::string &message) {
    vector<byte> resMsg;
    channels[i]->readWithSizeIntoVector(resMsg);
    const byte * uc = &(resMsg[0]);
    string resMsgStr(reinterpret_cast<char const*>(uc), resMsg.size());
    message = resMsgStr;
}

void Client::print_send_message(const string  &s) {
    logger(logger_out, "Sending message: %s\n", s.c_str());
}

void Client::print_recv_message(const string &s) {
    logger(logger_out, "Receiving message: %s\n", s.c_str());
}

void Client::print_local_data() {
    for (int i = 0; i < sample_num; i++) {
        std::cout<<"sample "<<i<<" = ";
        for (int j = 0; j < feature_num; j++) {
            if (j != feature_num - 1) {
                std::cout<<local_data[i][j]<<", ";
            } else {
                std::cout<<local_data[i][j]<<std::endl;
            }
        }
    }
}

void Client::print_labels() {
    if (!has_label) return;

    for (int i = 0; i < sample_num; i++) {
        std::cout<<"sample "<<i<<" label = "<<labels[i]<<std::endl;
    }
}

Client::~Client() {
    io_service.stop();
    djcs_t_free_public_key(m_pk);
    djcs_t_free_auth_server(m_au);
    hcs_free_random(m_hr);
}
