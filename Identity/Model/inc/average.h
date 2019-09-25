#ifndef AVERAGE_H
#define AVERAGE_H

#include <cmath>

#include "datatypes.h"

extern void movingAverage(float64_t *avg, const float64_t data, const float64_t coeff);
extern void movingSquareAverage(float64_t *sqr_avg, const float64_t data, const float64_t coeff);

#endif

