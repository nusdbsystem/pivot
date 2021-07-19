//
// Created by wuyuncheng on 12/10/19.
//

#include "encoder.h"
#include <iomanip>
#include <sstream>
#include <cmath>
#include <iostream>
extern FILE * logger_out;

EncodedNumber::EncodedNumber()
{
    mpz_init(n);
    mpz_init(value);
    type = Plaintext;
}

EncodedNumber::EncodedNumber(const EncodedNumber &number)
{
    mpz_init(n);
    mpz_init(value);
    mpz_set(n, number.n);
    mpz_set(value, number.value);
    exponent = number.exponent;
    type = number.type;
}

EncodedNumber& EncodedNumber::operator=(const EncodedNumber &number) {
    mpz_init(n);
    mpz_init(value);
    mpz_set(n, number.n);
    mpz_set(value, number.value);
    exponent = number.exponent;
    type = number.type;
}

void EncodedNumber::set_integer(mpz_t pn, int v)
{
    mpz_set(n, pn);
    fixed_pointed_encode(v, value, exponent);
    //set_float(pn, (float) v);
}

void EncodedNumber::set_float(mpz_t pn, float v, int precision)
{
    mpz_set(n, pn);
    fixed_pointed_encode(v, precision, value, exponent);
}

EncodedNumber::~EncodedNumber()
{
    //std::cout<<"destructor of EncodedNumber"<<std::endl;
    //gmp_printf("value = %Zd\n", value);
    mpz_clear(n);
    mpz_clear(value);
}

void EncodedNumber::decrease_exponent(int new_exponent)
{
    if (new_exponent > exponent) {
        logger(logger_out, "New exponent %d should be more negative"
                       "than old exponent %d", new_exponent, exponent);
        return;
    }
    if (new_exponent == exponent) return;
    unsigned long int exp_diff = exponent - new_exponent;
    mpz_t t;
    mpz_init(t);
    mpz_ui_pow_ui(t, 10, exp_diff);

    mpz_mul(value, value, t);
    exponent = new_exponent;
}

void EncodedNumber::increase_exponent(int new_exponent)
{
    if (new_exponent < exponent) {
        logger(logger_out, "New exponent %d should be more positive "
                       "than old exponent %d", new_exponent, exponent);
        return;
    }
    if (new_exponent == exponent) return;
    // 1. convert to string value
    char *t = mpz_get_str(NULL, 10, value);
    std::string s = t;
    // 2. truncate, should preserve the former (s_size - exp_diff) elements
    unsigned long int exp_diff = new_exponent - exponent;
    int v_size = s.size() - exp_diff;

    if (v_size <= 0) {
        logger(logger_out, "increase exponent error when truncating\n");
        return;
    }
    char *r = new char[v_size + 1];
    for (int j = 0; j < v_size; ++j) {
        r[j] = t[j];
    }
    r[v_size] = '\0';  // if miss this line, new_value would be any value unexpected
    // 3. convert back
    exponent = new_exponent;
    mpz_t new_value;
    mpz_init(new_value);
    mpz_set_str(new_value, r, 10);
    mpz_set(value, new_value);

    mpz_clear(new_value);
}

void EncodedNumber::decode(long &v)
{
    if (exponent != 0) {
        // not integer, should not call this decode function
        logger(logger_out, "exponent is not zero, failed, should call decode with float\n");
        return;
    }
    switch (check_encoded_number()) {
        case Positive:
            fixed_pointed_decode(v, value);
            break;
        case Negative:
            mpz_sub(value, value, n);
            fixed_pointed_decode(v, value);
            break;
        case Overflow:
            logger(logger_out, "encoded number %Zd is overflow\n", value);
            return;
        default:
            logger(logger_out, "encoded number %Zd is corrupted\n", value);
            return;
    }
}

void EncodedNumber::decode(float &v)
{
    switch (check_encoded_number()) {
        case Positive:
            fixed_pointed_decode(v, value, exponent);
            break;
        case Negative:
            mpz_sub(value, value, n);
            fixed_pointed_decode(v, value, exponent);
            break;
        case Overflow:
            logger(logger_out, "encoded number %Zd is overflow\n", value);
            break;
        default:
            logger(logger_out, "encoded number %Zd is corrupted\n", value);
            return;
    }
}

void EncodedNumber::decode_with_truncation(float &v, int truncated_exponent)
{
    switch (check_encoded_number()) {
        case Positive:
            fixed_pointed_decode_truncated(v, value, exponent, truncated_exponent);
            break;
        case Negative:
            mpz_sub(value, value, n);
            fixed_pointed_decode_truncated(v, value, exponent, truncated_exponent);
            break;
        case Overflow:
            logger(logger_out, "encoded number %Zd is overflow\n", value);
            break;
        default:
            logger(logger_out, "encoded number %Zd is corrupted\n", value);
            return;
    }
}

EncodedNumberState EncodedNumber::check_encoded_number()
{
    // max_int is the threshold of positive number
    // neg_int is the threshold of negative number
    mpz_t max_int, neg_int;
    mpz_init(max_int);
    compute_decode_threshold(max_int);
    mpz_init(neg_int);
    mpz_sub(neg_int, n, max_int);
    EncodedNumberState state;
    if (mpz_cmp(value, n) >= 0) {
        state = Invalid;
    } else if (mpz_cmp(value, max_int) <= 0) {
        state = Positive;
    } else if (mpz_cmp(value, neg_int) >= 0) {
        state = Negative;
    } else {
        state = Overflow;
    }

    mpz_clear(max_int);
    mpz_clear(neg_int);
    return state;
}

void EncodedNumber::compute_decode_threshold(mpz_t max_int)
{
    mpz_t t;
    mpz_init(t);
    mpz_fdiv_q_ui(t, n, 3);
    mpz_sub_ui(max_int, t, 1);  // this is max int
    mpz_clear(t);
}

long long fixed_pointed_integer_representation(float value, int precision){
    long long ex = (long long) pow(10, precision);
    std::stringstream ss;
    ss << std::fixed << std::setprecision(precision) << value;
    std::string s = ss.str();
    long long r = (long long) (::atof(s.c_str()) * ex);
    return r;
}

void fixed_pointed_encode(long value, mpz_t res, int & exponent) {
    mpz_set_si(res, value);
    exponent = 0;
}

void fixed_pointed_encode(float value, int precision, mpz_t res, int & exponent) {
    long long r = fixed_pointed_integer_representation(value, precision);
    mpz_set_si(res, r);
    exponent = 0 - precision;
}

void fixed_pointed_decode(long & value, mpz_t res) {
    value = mpz_get_si(res);
}

void fixed_pointed_decode(float & value, mpz_t res, int exponent) {
    if (exponent >= 0) {
        logger(logger_out, "decode mpz_t for float value failed\n");
        return;
    }
    if (exponent <= 0 - 2 * FLOAT_PRECISION) {
        fixed_pointed_decode_truncated(value, res, exponent, 0 - 2 * FLOAT_PRECISION);
    } else {
        char *t = mpz_get_str(NULL, 10, res);
        long v = ::atol(t);

        if (v == 0) {
            value = 0;
        } else {
            value = (float) (v * pow(10, exponent));
        }
        free(t);
    }
}


void fixed_pointed_decode_truncated(float & value, mpz_t res, int exponent, int truncated_exponent) {
    if (exponent >= 0 || truncated_exponent >= 0) {
        logger(logger_out, "decode mpz_t for float value failed\n");
        return;
    }
    int real_exponent = exponent >= truncated_exponent ? exponent : truncated_exponent;
    // convert to string and truncate string before assign to long
    char *t = mpz_get_str(NULL, 10, res);
    std::string s(t);
    // handle the case that the result value is 0
    if (::atol(t) == 0) {
        value = 0;
        return;
    }
    // should preserve the former (s_size - exponent + real_exponent) elements
    int v_size = s.size() + exponent - real_exponent;
    if (v_size <= 0) {
        logger(logger_out, "decode error when truncating to desired exponent\n");
        return;
    }
    char *r = new char[v_size + 1];
    for (int j = 0; j < v_size; ++j) {
        r[j] = t[j];
    }
    r[v_size] = '\0';
    long v = ::atol(r);
    value = (float) (v * pow(10, real_exponent));

    free(t);
    delete [] r;
}

void decrypt_temp(djcs_t_public_key *pk, djcs_t_auth_server **au, int required_client_num, EncodedNumber & rop, EncodedNumber v) {
    mpz_t t;
    mpz_init(t);
    rop.exponent = v.exponent;
    rop.type = Plaintext;
    mpz_set(rop.n, v.n);
    auto *dec = (mpz_t *) malloc (required_client_num * sizeof(mpz_t));
    for (int j = 0; j < required_client_num; j++) {
        mpz_init(dec[j]);
    }
    for (int j = 0; j < required_client_num; j++) {
        //djcs_t_share_decrypt(pk1, au[j], dec[j], ciphers[2].value);
        djcs_t_share_decrypt(pk, au[j], dec[j], v.value);
    }
    djcs_t_share_combine(pk, t, dec);
    mpz_set(rop.value, t);

    mpz_clear(t);
    for (int i = 0; i < required_client_num; i++) {
        mpz_clear(dec[i]);
    }
    free(dec);
}

void EncodedNumber::print_encoded_number() {
    logger(logger_out, "****** Print encoded number ******\n");
    char * n_str_c, * value_str_c;
    n_str_c = mpz_get_str(NULL, 10, n);
    value_str_c = mpz_get_str(NULL, 10, value);
    std::string n_str(n_str_c), value_str(value_str_c);
    logger(logger_out, "n = %s\n", n_str.c_str());
    logger(logger_out, "value = %s\n", value_str.c_str());
    logger(logger_out, "exponent = %d\n", exponent);
    logger(logger_out, "type = %d\n", type);
    logger(logger_out, "****** End print encoded number ******\n");
}