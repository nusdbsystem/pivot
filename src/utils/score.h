//
// Created by wuyuncheng on 10/1/20.
//

#ifndef PIVOT_SCORE_H
#define PIVOT_SCORE_H

#include <vector>

/**
 * given two vectors, compute the mean squared error (i.e., accuracy) in regression
 *
 * @param a
 * @param b
 * @return
 */
float mean_squared_error(std::vector<float> a, std::vector<float> b);

/**
 * given two values, compare whether they are same within some accuracy
 * TO compensate the accuracy loss of float values from SPDZ
 *
 * @param a
 * @param b
 * @return
 */
bool rounded_comparison(float a, float b);

/**
 * given an input float vector, compute the softmax probability distribution
 * FOR classification in GBDT
 *
 * @param inputs
 * @return
 */
std::vector<float> softmax(std::vector<float> inputs);

/**
 * return the argmax index of an input vector
 *
 * @param inputs
 * @return
 */
float argmax(std::vector<float> inputs);

#endif //PIVOT_SCORE_H
