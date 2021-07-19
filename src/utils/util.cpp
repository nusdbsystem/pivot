//
// Created by wuyuncheng on 11/10/19.
//

#include "util.h"
#include "encoder.h"
#include <cstdarg>
#include <ctime>
#include <cstdlib>
#include <string>

extern FILE * logger_out;

std::string get_timestamp_str() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(buffer,sizeof(buffer),"%d%m%Y%H%M%S",timeinfo);
    std::string str(buffer);
    return str;
}

void logger(FILE *out, const char *format, ...) {
    char buf[BUFSIZ] = {'\0'};
    char date_buf[50] = {'\0'};
    va_list ap;
    va_start(ap, format);
    vsprintf(buf, format, ap);
    va_end(ap);
    time_t current_time;
    current_time = time(NULL);
    struct tm *tm_struct = localtime(&current_time);
    sprintf(date_buf,"%04d-%02d-%02d %02d:%02d:%02d",
            tm_struct->tm_year + 1900,
            tm_struct->tm_mon + 1,
            tm_struct->tm_mday,
            tm_struct->tm_hour,
            tm_struct->tm_min,
            tm_struct->tm_sec);
    fprintf(out, "%s %s", date_buf, buf);
    fflush(out);
}

void print_string(const char *str){
    logger(logger_out, "%s", str);
}

void compute_thresholds(djcs_t_public_key *pk, mpz_t n, mpz_t positive_threshold, mpz_t negative_threshold) {
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