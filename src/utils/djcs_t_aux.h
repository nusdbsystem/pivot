//
// Created by wuyuncheng on 12/10/19.
//

#ifndef PIVOT_DJCS_T_AUX_H
#define PIVOT_DJCS_T_AUX_H

#include <vector>
#include "gmp.h"
#include "libhcs.h"
#include "encoder.h"

/**
 * encrypt an EncodedNumber and return an EncodedNumber
 *
 * @param pk
 * @param hr
 * @param res : result
 * @param plain
 */
void djcs_t_aux_encrypt(djcs_t_public_key* pk, hcs_random* hr, EncodedNumber & res, EncodedNumber plain);

/**
 * decrypt an EncodedNumber and return an EncodedNumber
 *
 * @param pk
 * @param au
 * @param res : result
 * @param cipher
 */
void djcs_t_aux_partial_decrypt(djcs_t_public_key* pk, djcs_t_auth_server* au,
        EncodedNumber res, EncodedNumber cipher);

/**
 * combine shares give hcs_shares, should set n and exponent
 * before calling this function
 *
 * @param pk
 * @param res
 * @param shares
 */
void djcs_t_aux_share_combine(djcs_t_public_key* pk, EncodedNumber res, mpz_t* shares);

/**
 * homomorphic addition of two ciphers and return an EncodedNumber
 *
 * @param pk
 * @param res : res
 * @param cipher1
 * @param cipher2
 */
void djcs_t_aux_ee_add(djcs_t_public_key* pk, EncodedNumber & res, EncodedNumber cipher1, EncodedNumber cipher2);

/**
 * homomorphic multiplication of a cipher and a plain, return an EncodedNumber
 *
 * @param pk
 * @param res : res
 * @param cipher
 * @param plain
 */
void djcs_t_aux_ep_mul(djcs_t_public_key* pk, EncodedNumber & res, EncodedNumber cipher, EncodedNumber plain);

/**
 * homomorphic inner product of a cipher vector and a plain vector
 * return an EncodedNumber
 *
 * @param pk
 * @param hr
 * @param res
 * @param ciphers
 * @param plains
 * @param size
 */
void djcs_t_aux_inner_product(djcs_t_public_key* pk, hcs_random* hr, EncodedNumber & res, EncodedNumber ciphers[], EncodedNumber plains[], int size);

#endif //PIVOT_DJCS_T_AUX_H
