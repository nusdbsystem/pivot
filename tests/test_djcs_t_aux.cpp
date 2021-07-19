//
// Created by wuyuncheng on 15/10/19.
//

#include "../src/utils/djcs_t_aux.h"
#include "../src/utils/util.h"
#include "../src/utils/encoder.h"
#include "libhcs.h"
#include "gmp.h"
#include <cmath>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "test_djcs_t_aux.h"

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

void aux_compute_thresholds() {
    mpz_t g;
    mpz_init(g);
    mpz_set(g, pk->g);
    mpz_sub_ui(n, g, 1);
    mpz_t t;
    mpz_init(t);
    mpz_fdiv_q_ui(t, n, 3);
    mpz_sub_ui(positive_threshold, t, 1);  // this is positive threshold
    mpz_sub(negative_threshold, n, positive_threshold);  // this is negative threshold

    mpz_clear(g);
    mpz_clear(t);
}

void test_encryption_decryption_int(int x) {
    EncodedNumber *a = new EncodedNumber();
    a->set_integer(n, x);
    EncodedNumber *encrypted_a = new EncodedNumber();
    djcs_t_aux_encrypt(pk, hr, *encrypted_a, *a);
    EncodedNumber *decrypted_a = new EncodedNumber();
    decrypted_a->exponent = encrypted_a->exponent;
    mpz_set(decrypted_a->n, encrypted_a->n);
    mpz_t *dec = (mpz_t *) malloc (REQUIRED_CLIENT_DECRYPTION * sizeof(mpz_t));
    for (int j = 0; j < REQUIRED_CLIENT_DECRYPTION; j++) {
        mpz_init(dec[j]);
    }
    for (int j = 0; j < REQUIRED_CLIENT_DECRYPTION; j++) {
        djcs_t_share_decrypt(pk, au[j], dec[j], encrypted_a->value);
    }
    djcs_t_share_combine(pk, decrypted_a->value, dec);
    decrypted_a->type = Plaintext;

    // decode decrypted_a
    float y;
    decrypted_a->decode(y);
    if ((float) x != y) {
        logger(logger_out, "test_encryption_decryption_int: "
                       "the decrypted plaintext %f is not match original plaintext %f, failed\n", y, x);
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_encryption_decryption_int；"
                       "the encryption and decryption test succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
}

void test_encryption_decryption_float(float x) {
    auto *a = new EncodedNumber();
    a->set_float(n, x, FLOAT_PRECISION);
    auto *encrypted_a = new EncodedNumber();
    djcs_t_aux_encrypt(pk, hr, *encrypted_a, *a);
    auto *decrypted_a = new EncodedNumber();
    decrypted_a->exponent = encrypted_a->exponent;
    mpz_set(decrypted_a->n, encrypted_a->n);
    auto *dec = (mpz_t *) malloc (REQUIRED_CLIENT_DECRYPTION * sizeof(mpz_t));
    for (int j = 0; j < REQUIRED_CLIENT_DECRYPTION; j++) {
        mpz_init(dec[j]);
    }
    for (int j = 0; j < REQUIRED_CLIENT_DECRYPTION; j++) {
        djcs_t_share_decrypt(pk, au[j], dec[j], encrypted_a->value);
    }
    djcs_t_share_combine(pk, decrypted_a->value, dec);
    decrypted_a->type = Plaintext;

    // decode decrypted_a
    float y;
    decrypted_a->decode(y);
    if (fabs(y - x) >= PRECISION_THRESHOLD) {
        logger(logger_out, "test_encryption_decryption_float: "
                       "the decrypted plaintext %f is not match original plaintext %f, failed\n", y, x);
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_encryption_decryption_float: "
                       "the encryption and decryption test succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
}

void test_ee_add() {
    auto *plain1 = new EncodedNumber();
    //plain1->set_float(n1, 0.25, FLOAT_PRECISION);
    plain1->set_integer(n, 10);
    auto *cipher1 = new EncodedNumber();
    djcs_t_aux_encrypt(pk, hr, *cipher1, *plain1);
    auto *plain2 = new EncodedNumber();
    plain2->set_float(n, 0.5, FLOAT_PRECISION);
    auto *cipher2 = new EncodedNumber();
    djcs_t_aux_encrypt(pk, hr, *cipher2, *plain2);
    auto *res = new EncodedNumber();
    djcs_t_aux_ee_add(pk, *res, *cipher1, *cipher2);
    auto *decryption = new EncodedNumber();
    decryption->exponent = res->exponent;
    mpz_set(decryption->n, res->n);
    auto *dec = (mpz_t *) malloc (REQUIRED_CLIENT_DECRYPTION * sizeof(mpz_t));
    for (int j = 0; j < REQUIRED_CLIENT_DECRYPTION; j++) {
        mpz_init(dec[j]);
    }
    for (int j = 0; j < REQUIRED_CLIENT_DECRYPTION; j++) {
        djcs_t_share_decrypt(pk, au[j], dec[j], res->value);
    }
    djcs_t_share_combine(pk, decryption->value, dec);
    decryption->type = Plaintext;

    // decode decrypted_a
    float y;
    decryption->decode(y);
    if (fabs(y - (10 + 0.5)) >= PRECISION_THRESHOLD) {
        logger(logger_out, "test_ee_add: "
                       "the decrypted result %f is not match original plaintext addition %f, failed\n", y, 10 + 0.5);
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_ee_add: "
                       "the homomorphic addition computation test succeed, expected value = %f, real value = %f\n", 10 + 0.5, y);
        total_cases_num += 1;
        passed_cases_num += 1;
    }
}

void test_ep_mul() {
    auto *plain1 = new EncodedNumber();
    plain1->set_float(n, 0.25, FLOAT_PRECISION);
    auto *cipher1 = new EncodedNumber();
    djcs_t_aux_encrypt(pk, hr, *cipher1, *plain1);
    auto *plain2 = new EncodedNumber();
    plain2->set_float(n, 0.12, FLOAT_PRECISION);
    auto *res = new EncodedNumber();
    djcs_t_aux_ep_mul(pk, *res, *cipher1, *plain2);
    auto *decryption = new EncodedNumber();
    decryption->exponent = res->exponent;
    mpz_set(decryption->n, res->n);
    auto *dec = (mpz_t *) malloc (REQUIRED_CLIENT_DECRYPTION * sizeof(mpz_t));
    for (int j = 0; j < REQUIRED_CLIENT_DECRYPTION; j++) {
        mpz_init(dec[j]);
    }
    for (int j = 0; j < REQUIRED_CLIENT_DECRYPTION; j++) {
        djcs_t_share_decrypt(pk, au[j], dec[j], res->value);
    }
    djcs_t_share_combine(pk, decryption->value, dec);
    decryption->type = Plaintext;
    // decode decrypted_a
    float y;
    decryption->decode(y);
    if (fabs(y - (0.25 * 0.12)) >= PRECISION_THRESHOLD) {
        logger(logger_out, "test_ep_mul: "
                       "the decrypted result %f is not match original plaintext addition %f, failed\n", y, 0.25 + 0.12);
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_ep_mul: "
                       "the homomorphic multiplication computation test succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
}

void test_inner_product_int() {
    int feature_num = 3;
    float plain_res = 0.0;
    EncodedNumber *ciphers = new EncodedNumber[feature_num];
    EncodedNumber *plains = new EncodedNumber[feature_num];
    for (int i = 0; i < feature_num; i++) {
        EncodedNumber *plain1 = new EncodedNumber();
        plain1->set_integer(n, i + 1);
        djcs_t_aux_encrypt(pk, hr, ciphers[i], *plain1);
        plains[i].set_integer(n, feature_num + i);
        plain_res = plain_res + (i + 1) * (feature_num + i);
    }

    EncodedNumber res;
    djcs_t_aux_inner_product(pk, hr, res, ciphers, plains, feature_num);
    auto *decryption = new EncodedNumber();
    decryption->exponent = res.exponent;
    mpz_set(decryption->n, res.n);
    auto *dec = (mpz_t *) malloc (REQUIRED_CLIENT_DECRYPTION * sizeof(mpz_t));
    for (int j = 0; j < REQUIRED_CLIENT_DECRYPTION; j++) {
        mpz_init(dec[j]);
    }
    for (int j = 0; j < REQUIRED_CLIENT_DECRYPTION; j++) {
        //djcs_t_share_decrypt(pk1, au[j], dec[j], ciphers[2].value);
        djcs_t_share_decrypt(pk, au[j], dec[j], res.value);
    }
    djcs_t_share_combine(pk, decryption->value, dec);
    decryption->type = Plaintext;

    // decode decrypted_a
    float y;
    decryption->decode(y);
    if (fabs((float) y - plain_res) >= PRECISION_THRESHOLD) {
        logger(logger_out, "test_inner_product_int: "
                       "the decrypted result %f is not match original plaintext inner product %f, failed\n", y, plain_res);
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_inner_product_int: "
                       "the homomorphic inner product computation test succeed, expected value = %f, real value = %f\n", plain_res, y);
        total_cases_num += 1;
        passed_cases_num += 1;
    }
    delete [] ciphers;
    delete [] plains;
}

void test_inner_product_float() {
    int feature_num = 5;
    float plain_res = 0.0;
    EncodedNumber *ciphers = new EncodedNumber[feature_num];
    EncodedNumber *plains = new EncodedNumber[feature_num];
    for (int i = 0; i < feature_num; i++) {
        //ciphers[i] = new EncodedNumber();
        EncodedNumber *plain1 = new EncodedNumber();
        logger(logger_out, "the plain1 value = %f\n", ((float) i + 1) / ((float) feature_num + i));
        plain1->set_float(n, ((float) i + 1) / ((float) feature_num + i), FLOAT_PRECISION);
        djcs_t_aux_encrypt(pk, hr, ciphers[i], *plain1);
        plains[i].set_float(n, ((float) i + 2) / ((float) feature_num + i), 2 * FLOAT_PRECISION);
        plain_res = plain_res + ((float) i + 1) / ((float) feature_num + i) * ((float) i + 2) / ((float) feature_num + i);
    }
    EncodedNumber res;
    djcs_t_aux_inner_product(pk, hr, res, ciphers, plains, feature_num);
    auto *decryption = new EncodedNumber();
    decryption->exponent = res.exponent;
    mpz_set(decryption->n, res.n);
    auto *dec = (mpz_t *) malloc (REQUIRED_CLIENT_DECRYPTION * sizeof(mpz_t));
    for (int j = 0; j < REQUIRED_CLIENT_DECRYPTION; j++) {
        mpz_init(dec[j]);
    }
    for (int j = 0; j < REQUIRED_CLIENT_DECRYPTION; j++) {
        //djcs_t_share_decrypt(pk1, au[j], dec[j], ciphers[2].value);
        djcs_t_share_decrypt(pk, au[j], dec[j], res.value);
    }
    djcs_t_share_combine(pk, decryption->value, dec);
    decryption->type = Plaintext;

    // decode decrypted_a
    float y = 0.1;
    decryption->decode(y);
    if (fabs(y - plain_res) >= PRECISION_THRESHOLD) {
        logger(logger_out, "test_inner_product_float: "
                       "the decrypted result %f is not match original plaintext inner product %f, failed\n", y, plain_res);
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_inner_product_float: "
                       "the homomorphic inner product computation test succeed, expected_value = %f, real_value = %f\n", plain_res, y);
        total_cases_num += 1;
        passed_cases_num += 1;
    }
    delete [] ciphers;
    delete [] plains;
}

int test_djcs_t_aux()
{
    logger(logger_out, "****** Test djcs_t auxiliary functions ******\n");
//    hr = hcs_init_random();
//    pk = djcs_t_init_public_key();
//    vk = djcs_t_init_private_key();
//
//    djcs_t_generate_key_pair(pk, vk, hr, 1, 1024, 3, 3);
//
//    mpz_t *coeff = djcs_t_init_polynomial(vk, hr);
//
//    for (int i = 0; i < TOTAL_CLIENT_NUM; i++) {
//        mpz_init(si[i]);
//        djcs_t_compute_polynomial(vk, coeff, si[i], i);
//        au[i] = djcs_t_init_auth_server();
//        djcs_t_set_auth_server(au[i], si[i], i);
//    }
//
//    mpz_init(n);
//    mpz_init(positive_threshold);
//    mpz_init(negative_threshold);
    total_cases_num = 0;
    passed_cases_num = 0;
    // compute threshold
    // aux_compute_thresholds();
    // test encryption decryption int
    test_encryption_decryption_int(10);
    test_encryption_decryption_int(-10);  // not sure why call two times cause double free exception
    // test encryption decryption float
    test_encryption_decryption_float(0.123456);
    test_encryption_decryption_float(-0.654321);
    // test homomorphic addition
    test_ee_add();
    // test homomorphic multiplication
    test_ep_mul();
    // test homomorphic inner product int
    test_inner_product_int();
    // test homomorphic inner product float
    test_inner_product_float();
    logger(logger_out, "****** total_cases_num = %d, passed_cases_num = %d ******\n",
           total_cases_num, passed_cases_num);
    // free memory
//    hcs_free_random(hr);
//    djcs_t_free_public_key(pk);
//    djcs_t_free_private_key(vk);
//
//    mpz_clear(n);
//    mpz_clear(positive_threshold);
//    mpz_clear(negative_threshold);
//
//    for (int i = 0; i < TOTAL_CLIENT_NUM; i++) {
//        mpz_clear(si[i]);
//        djcs_t_free_auth_server(au[i]);
//    }
//    free(si);
//    free(au);
//
//    djcs_t_free_polynomial(vk, coeff);
    return 0;
}