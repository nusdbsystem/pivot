//
// Created by wuyuncheng on 11/10/19.
//

#ifndef PIVOT_UTIL_H
#define PIVOT_UTIL_H

#include <stdio.h>
#include "gmp.h"
#include "libhcs.h"
#include <string>
#include "../include/common.h"


#define REQUIRED_CLIENT_DECRYPTION 3 // for test

/**
 * get timestamp string
 * @return
 */
std::string get_timestamp_str();

/**
 * log file
 *
 * @param out
 * @param format
 * @param ...
 */
void logger(FILE* out, const char *format, ...);

/**
 * stdout print for debug
 *
 * @param str
 */
void print_string(const char *str);

void compute_thresholds(djcs_t_public_key *pk, mpz_t n, mpz_t positive_threshold, mpz_t negative_threshold);


#endif //PIVOT_UTIL_H
