
#include "average.h"


void movingAverage(float64_t *avg, const float64_t data, const float64_t coeff) {
	*avg = (coeff * (*avg)) + ((1 - coeff) * data);
}

void movingSquareAverage(float64_t *sqr_avg, const float64_t data, const float64_t coeff) {
	*sqr_avg = (coeff * (*sqr_avg)) + ((1 - coeff) * std::pow(data, 2));
}