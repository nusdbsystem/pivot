//
// Created by wuyuncheng on 12/10/19.
//


#include "djcs_t_aux.h"
#include <stdio.h>
#include <cstdlib>
extern FILE * logger_out;

void djcs_t_aux_encrypt(djcs_t_public_key* pk, hcs_random* hr, EncodedNumber & res, EncodedNumber plain) {
    if (plain.type != Plaintext) {
        logger(logger_out, "the plain should not be encrypted\n");
        return;
    }
    mpz_t t1, t2;
    mpz_init(t1);
    mpz_init(t2);
    mpz_set(t1, plain.value);
    djcs_t_encrypt(pk, hr, t2, t1);
    mpz_set(res.n, plain.n);
    mpz_set(res.value, t2);
    res.exponent = plain.exponent;
    res.type = Ciphertext;

    mpz_clear(t1);
    mpz_clear(t2);
}

void djcs_t_aux_partial_decrypt(djcs_t_public_key* pk, djcs_t_auth_server* au, EncodedNumber res, EncodedNumber cipher) {
    if (cipher.type != Ciphertext) {
        logger(logger_out, "the cipher should be encrypted\n");
        return;
    }
    mpz_set(res.n, cipher.n);
    res.exponent = cipher.exponent;
    djcs_t_share_decrypt(pk, au, res.value, cipher.value);
    res.type = Plaintext;
}

void djcs_t_aux_share_combine(djcs_t_public_key* pk, EncodedNumber res, mpz_t* shares) {
    djcs_t_share_combine(pk, res.value, shares);
    res.type = Plaintext;
}

void djcs_t_aux_ee_add(djcs_t_public_key* pk, EncodedNumber & res, EncodedNumber cipher1, EncodedNumber cipher2) {
    if (cipher1.type != Ciphertext || cipher2.type != Ciphertext) {
        logger(logger_out, "the two inputs should be ciphertexts\n");
        return;
    }
    if (mpz_cmp(cipher1.n, cipher2.n) != 0) {
        logger(logger_out, "two ciphertexts not with the same public key\n");
        return;
    }
    if (cipher1.exponent != cipher2.exponent) {
        logger(logger_out, "two ciphertexts do not have the same exponents, meaningless addtion\n");
        return;
    }
    mpz_t t;
    mpz_init(t);
    djcs_t_ee_add(pk, t, cipher1.value, cipher2.value);
    mpz_set(res.value, t);
    mpz_set(res.n, cipher1.n);
    res.type = Ciphertext;
    res.exponent = cipher1.exponent;

    mpz_clear(t);
}

void djcs_t_aux_ep_mul(djcs_t_public_key* pk, EncodedNumber & res, EncodedNumber cipher, EncodedNumber plain) {
    if (cipher.type != Ciphertext || plain.type != Plaintext) {
        logger(logger_out, "the cipher should be encrypted or the plain should not be encrypted\n");
        return;
    }
    if (mpz_cmp(cipher.n, plain.n) != 0) {
        logger(logger_out, "two values not under the same public key\n");
        return;
    }
    mpz_set(res.n, cipher.n);
    res.exponent = cipher.exponent + plain.exponent;
    djcs_t_ep_mul(pk, res.value, cipher.value, plain.value);
    res.type = Ciphertext;
}

void djcs_t_aux_inner_product(djcs_t_public_key* pk, hcs_random* hr, EncodedNumber & res,
        EncodedNumber ciphers[], EncodedNumber plains[], int size) {
    mpz_set(res.n, ciphers[0].n);
    res.exponent = ciphers[0].exponent + plains[0].exponent;
    res.type = Ciphertext;
    // assume the elements in the plains have the same exponent, so does ciphers
    mpz_t *mpz_ciphers = (mpz_t *) malloc (size * sizeof(mpz_t));
    mpz_t *mpz_plains = (mpz_t *) malloc (size * sizeof(mpz_t));
    for (int i = 0; i < size; i++) {
        mpz_init(mpz_ciphers[i]);
        mpz_init(mpz_plains[i]);
        mpz_set(mpz_ciphers[i], ciphers[i].value);
        mpz_set(mpz_plains[i], plains[i].value);
    }
    // homomorphic dot product
    mpz_t sum, sum1;   // why need another sum1 that can only be correct? weired
    mpz_init(sum);
    mpz_init(sum1);
    mpz_set_si(sum1, 0);
    djcs_t_encrypt(pk, hr, sum, sum1);
    for (int j = 0; j < size; j++) {
        // store the encrypted_exact_answer
        mpz_t tmp;
        mpz_init(tmp);
        mpz_set_si(tmp, 0);
        djcs_t_encrypt(pk, hr, tmp, tmp);   // why here do not need another tmp1?
        djcs_t_ep_mul(pk, tmp, mpz_ciphers[j], mpz_plains[j]);
        djcs_t_ee_add(pk, sum, sum, tmp);
        mpz_clear(tmp);
    }
    mpz_set(res.value, sum);

    mpz_clear(sum);
    mpz_clear(sum1);
    for (int i = 0; i < size; i++) {
        mpz_clear(mpz_ciphers[i]);
        mpz_clear(mpz_plains[i]);
    }
    free(mpz_ciphers);
    free(mpz_plains);
}