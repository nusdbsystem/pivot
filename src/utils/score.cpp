//
// Created by wuyuncheng on 10/1/20.
//

#include <cmath>
#include "score.h"
#include "util.h"
#include "../include/common.h"
extern FILE * logger_out;

float mean_squared_error(std::vector<float> a, std::vector<float> b) {
    if (a.size() != b.size()) {
        logger(logger_out, "Mean squared error computation wrong: sizes of the two vectors not same\n");
    }
    int num = a.size();
    float squared_error = 0.0;
    for (int i = 0; i < num; i++) {
        squared_error = squared_error + (a[i] - b[i]) * (a[i] - b[i]);
    }
    float mean_squared_error = squared_error / num;
    return mean_squared_error;
}

bool rounded_comparison(float a, float b) {
    if ((a >= b - ROUNDED_PRECISION) && (a <= b + ROUNDED_PRECISION)) return true;
    else return false;
}

std::vector<float> softmax(std::vector<float> inputs) {
    float sum = 0.0;
    for (int i = 0; i < inputs.size(); i++) {
        sum += inputs[i];
    }
    std::vector<float> probs;
    for (int i = 0; i < inputs.size(); i++) {
        probs.push_back(inputs[i]/sum);
    }
    return probs;
}

float argmax(std::vector<float> inputs) {
    float index = 0, max = -1;
    for (int i = 0; i < inputs.size(); i++) {
        if (max < inputs[i]) {
            max = inputs[i];
            index = i;
        }
    }
    return index;
}