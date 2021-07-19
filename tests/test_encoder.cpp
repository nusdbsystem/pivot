//
// Created by wuyuncheng on 15/10/19.
//

#include <iostream>
#include "../src/utils/encoder.h"
#include "libhcs.h"
#include "../src/utils/util.h"
#include <iomanip>
#include <cmath>
#include <sstream>
#include "test_encoder.h"

extern hcs_random *hr;
extern djcs_t_public_key *pk;
extern djcs_t_private_key *vk;
extern mpz_t n, positive_threshold, negative_threshold;
extern FILE * logger_out;
int total_cases_num, passed_cases_num;

//void compute_thresholds() {
//    mpz_t g;
//    mpz_init(g);
//    mpz_set(g, pk->g);
//    mpz_sub_ui(n, g, 1);
//
//    mpz_t t;
//    mpz_init(t);
//    mpz_fdiv_q_ui(t, n, 3);
//    mpz_sub_ui(positive_threshold, t, 1);  // this is positive threshold
//    mpz_sub(negative_threshold, n, positive_threshold);  // this is negative threshold
//
//    mpz_clear(g);
//    mpz_clear(t);
//}

void test_positive_int(int x) {
    EncodedNumber a;
    float y;
    a.set_integer(n, x);
    a.decode(y);
    if ((float) x != y) {
        logger(logger_out, "test_positive_int(%d) failed\n", x);
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_positive_int(%d) succeed\n", x);
        total_cases_num += 1;
        passed_cases_num += 1;
    }
}

void test_negative_int(int x) {
    EncodedNumber a;
    float y;
    a.set_integer(n, x);
    a.decode(y);
    if ((float) x != y) {
        logger(logger_out, "test_negative_int(%d) failed\n", x);
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_negative_int(%d) succeed\n", x);
        total_cases_num += 1;
        passed_cases_num += 1;
    }
}

void test_positive_float(float x) {
    EncodedNumber a;
    float y;
    a.set_float(n, x, FLOAT_PRECISION);
    a.decode(y);
    if ( x != y) {
        logger(logger_out, "test_positive_float(%f) failed, decoded value = %d\n", x, y);
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_positive_float(%f) succeed\n", x);
        total_cases_num += 1;
        passed_cases_num += 1;
    }
}

void test_negative_float(float x) {
    EncodedNumber a;
    float y;
    a.set_float(n, x);
    a.decode(y);
    if ( x != y) {
        logger(logger_out, "test_negative_float(%f) failed, decoded value = %f\n", x, y);
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_negative_float(%f) succeed\n", x);
        total_cases_num += 1;
        passed_cases_num += 1;
    }
}

void test_encoded_number_state() {
    EncodedNumber a, b, c, d;
    mpz_t t1, t2, t3, t4;
    mpz_init(t1);
    mpz_init(t2);
    mpz_init(t3);
    mpz_init(t4);
    mpz_set(a.n, n);
    mpz_set(t1, positive_threshold);
    mpz_sub_ui(t1, t1, 1);
    mpz_set(a.value, t1);
    a.exponent = 0;
    if (a.check_encoded_number() != Positive) {
        logger(logger_out, "test_encoded_number_state Positive failed\n");
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_encoded_number_state Positive succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
    mpz_set(b.n, n);
    mpz_set(t2, negative_threshold);
    mpz_add_ui(t2, t2, 1);
    mpz_set(b.value, t2);
    b.exponent = 0;
    if (b.check_encoded_number() != Negative) {
        logger(logger_out, "test_encoded_number_state Negative failed\n");
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_encoded_number_state Negative succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
    mpz_set(c.n, n);
    mpz_set(t3, positive_threshold);
    mpz_add_ui(t3, t3, 1);
    mpz_set(c.value, t3);
    c.exponent = 0;
    if (c.check_encoded_number() != Overflow) {
        logger(logger_out, "test_encoded_number_state Overflow failed\n");
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_encoded_number_state Overflow succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
    mpz_set(d.n, n);
    mpz_set(t4, n);
    mpz_add_ui(t4, t4, 1);
    mpz_set(d.value, t4);
    d.exponent = 0;
    if (d.check_encoded_number() != Invalid) {
        logger(logger_out, "test_encoded_number_state Invalid failed\n");
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_encoded_number_state Invalid succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
    mpz_clear(t1);
    mpz_clear(t2);
    mpz_clear(t3);
    mpz_clear(t4);
}

void test_decrease_exponent_positive_int() {
    EncodedNumber a;
    a.set_integer(n, 1);
    a.decrease_exponent(0 - FLOAT_PRECISION);
    mpz_t t;
    mpz_init(t);
    long x = (long) pow(10, FLOAT_PRECISION);
    std::string str;
    std::stringstream str_stream;
    str_stream << x;
    str_stream >> str;
    mpz_set_str(t, str.c_str(), 10);
    if ( (mpz_cmp(t, a.value) != 0) || (a.exponent != 0 - FLOAT_PRECISION)) {
        logger(logger_out, "test_decrease_exponent_positive_int failed\n");
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_decrease_exponent_positive_int succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
    mpz_clear(t);
}

void test_decrease_exponent_negative_int() {
    EncodedNumber a;
    a.set_integer(n, -5);
    a.decrease_exponent(0 - FLOAT_PRECISION);
    mpz_t t;
    mpz_init(t);
    long x = -5 * (long) pow(10, FLOAT_PRECISION);
    std::string str;
    std::stringstream str_stream;
    str_stream << x;
    str_stream >> str;
    mpz_set_str(t, str.c_str(), 10);
    if ( (mpz_cmp(t, a.value) != 0) || (a.exponent != 0 - FLOAT_PRECISION)) {
        logger(logger_out, "test_decrease_exponent_negative_int failed\n");
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_decrease_exponent_negative_int succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
    mpz_clear(t);
}

void test_decrease_exponent_positive_float() {
    EncodedNumber a;
    a.set_float(n, 0.123456, FLOAT_PRECISION);
    a.decrease_exponent(0 - 2 * FLOAT_PRECISION);
    mpz_t t;
    mpz_init(t);
    long x = 0.123456 * (long) pow(10, 2 * FLOAT_PRECISION);
    std::string str;
    std::stringstream str_stream;
    str_stream << x;
    str_stream >> str;
    mpz_set_str(t, str.c_str(), 10);
    if ( (mpz_cmp(t, a.value) != 0) || (a.exponent != 0 - 2 * FLOAT_PRECISION)) {
        logger(logger_out, "test_decrease_exponent_positive_float failed\n");
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_decrease_exponent_positive_float succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
    mpz_clear(t);
}

void test_decrease_exponent_negative_float() {
    EncodedNumber a;
    //a.set_float(n, -0.654321, FLOAT_PRECISION); // will fail due to float representation
    a.set_float(n, -0.000005, FLOAT_PRECISION);
    a.decrease_exponent(0 - 2 * FLOAT_PRECISION);
    mpz_t t;
    mpz_init(t);
    long x = -0.000005 * (long) pow(10, 2 * FLOAT_PRECISION);
    std::string str;
    std::stringstream str_stream;
    str_stream << x;
    str_stream >> str;
    mpz_set_str(t, str.c_str(), 10);

    // NOTE: this test should not succeed, because -0.654321 will be represented as
    // -0.654321015 in the bit vector form, thus there is some precision problem,
    // similar problem occurs to other float representation (it depends). A better
    // way to test this function is to decode and compare.

    if ( (mpz_cmp(t, a.value) != 0) || (a.exponent != 0 - 2 * FLOAT_PRECISION)) {
        logger(logger_out, "test_decrease_exponent_negative_float failed\n");
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_decrease_exponent_negative_float succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
    mpz_clear(t);
}

void test_increase_exponent_positive_float() {
    EncodedNumber a;
    a.set_float(n, 0.0000000105, 2 * FLOAT_PRECISION);
    a.increase_exponent(0 - FLOAT_PRECISION);
    mpz_t t;
    mpz_init(t);
    long x = 0.00000001 * (long) pow(10, FLOAT_PRECISION);
    std::string str;
    std::stringstream str_stream;
    str_stream << x;
    str_stream >> str;
    mpz_set_str(t, str.c_str(), 10);
    if ( (mpz_cmp(t, a.value) != 0) || (a.exponent != 0 - FLOAT_PRECISION)) {
        logger(logger_out, "test_increase_exponent_positive_float failed\n");
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_increase_exponent_positive_float succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
    mpz_clear(t);
}

void test_increase_exponent_negative_float() {
    EncodedNumber a;
    a.set_float(n, -0.0000000105, 2 * FLOAT_PRECISION);
    a.increase_exponent(0 - FLOAT_PRECISION);
    mpz_t t;
    mpz_init(t);
    long x = -0.00000001 * (long) pow(10, FLOAT_PRECISION);
    std::string str;
    std::stringstream str_stream;
    str_stream << x;
    str_stream >> str;
    mpz_set_str(t, str.c_str(), 10);

    // NOTE: this test should not succeed, because -0.654321 will be represented as
    // -0.654321015 in the bit vector form, thus there is some precision problem,
    // similar problem occurs to other float representation (it depends). A better
    // way to test this function is to decode and compare.

    if ( (mpz_cmp(t, a.value) != 0) || (a.exponent != 0 - FLOAT_PRECISION)) {
        logger(logger_out, "test_increase_exponent_negative_float failed\n");
        total_cases_num += 1;
    } else {
        logger(logger_out, "test_increase_exponent_negative_float succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
    mpz_clear(t);
}

void test_decode_with_truncation_float() {
    EncodedNumber a;
    a.set_float(n, -0.0000000105, 2 * FLOAT_PRECISION);
    float x;
    a.decode_with_truncation(x, 0 - FLOAT_PRECISION);
    if (fabs(x + 0.00000001) >= PRECISION_THRESHOLD) {
        logger(logger_out, "decode with truncation failed\n");
        total_cases_num += 1;
    } else {
        logger(logger_out, "decode with truncation succeed\n");
        total_cases_num += 1;
        passed_cases_num += 1;
    }
}

int test_encoder()
{
    logger(logger_out, "****** Test encode and decode functions ******\n");
//    hr = hcs_init_random();
//    pk = djcs_t_init_public_key();
//    vk = djcs_t_init_private_key();
//    djcs_t_generate_key_pair(pk, vk, hr, 1, 512, 3, 3);
//    mpz_init(n);
//    mpz_init(positive_threshold);
//    mpz_init(negative_threshold);
    total_cases_num = 0;
    passed_cases_num = 0;
    // compute thresholds
    // compute_thresholds();
    // 1. test positive int
    test_positive_int(5);
    // 2. test negative int
    test_negative_int(-5);
    // 3. test positive float
    test_positive_float(0.123456);
    // 4. test negative float
    test_negative_float(-0.654321);
    // 5. test encoded number state
    test_encoded_number_state();
    // 6. test decrease exponent positive int
    test_decrease_exponent_positive_int();
    // 7. test decrease exponent negative int
    test_decrease_exponent_negative_int();
    // 8. test decrease exponent positive float
    test_decrease_exponent_positive_float();
    // 9. test decrease exponent negative float
    test_decrease_exponent_negative_float();
    // 10. test increase exponent positive float
    test_increase_exponent_positive_float();
    // 11. test increase exponent negative float
    test_increase_exponent_negative_float();
    // 12. test decode with truncation positive float
    test_decode_with_truncation_float();
    logger(logger_out, "****** total_cases_num = %d, passed_cases_num = %d ******\n",
            total_cases_num, passed_cases_num);
    // free memory
//    hcs_free_random(hr);
//    djcs_t_free_public_key(pk);
//    djcs_t_free_private_key(vk);
//    mpz_clear(n);
//    mpz_clear(positive_threshold);
//    mpz_clear(negative_threshold);
    return 0;
}
