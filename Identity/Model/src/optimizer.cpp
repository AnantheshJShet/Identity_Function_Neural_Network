#include "optimizer.h"
#include "average.h"

Optimizer::Optimizer() {
	dL_dw_velocity = 0;
	dL_db_velocity = 0;
	avg_dL_dw = 0;
	avg_dL_db = 0;
	square_avg_dL_dw = 0;
	square_avg_dL_db = 0;
}

void Optimizer::init(const Optimizer_Method_T in_method,
					 const float64_t in_eta,
					 const float64_t in_gamma,
					 const float64_t in_beta1,
					 const float64_t in_beta2) {
	dL_dw_velocity = 0;
	dL_db_velocity = 0;
	avg_dL_dw = 0;
	avg_dL_db = 0;
	square_avg_dL_dw = 0;
	square_avg_dL_db = 0;

	method = in_method;

	eta = in_eta;
	gamma = in_gamma;
	beta1 = in_beta1;
	beta2 = in_beta2;
}

void Optimizer::reset() {
	dL_dw_velocity = 0;
	dL_db_velocity = 0;
	avg_dL_dw = 0;
	avg_dL_db = 0;
	square_avg_dL_dw = 0;
	square_avg_dL_db = 0;
}

void Optimizer::normal(float64_t &weight, float64_t &bias, const float64_t dL_dw, const float64_t dL_db) {
	weight -= (eta * dL_dw);
	bias -= (eta * dL_db);
}

void Optimizer::momentum(float64_t &weight, float64_t &bias, const float64_t dL_dw, const float64_t dL_db) {
#if 0
	dL_dw_velocity = gamma * dL_dw_velocity + eta*dL_dw;
	dL_db_velocity = gamma * dL_db_velocity + eta*dL_db;
	*weight -= dL_dw_velocity;
	*bias -= dL_db_velocity;
#else
	dL_dw_velocity = gamma * dL_dw_velocity + dL_dw;
	dL_db_velocity = gamma * dL_db_velocity + dL_db;
	weight -= (eta*dL_dw_velocity);
	bias -= (eta*dL_db_velocity);
#endif
}

void Optimizer::rms(float64_t &weight, float64_t &bias, const float64_t dL_dw, const float64_t dL_db) {
	movingSquareAverage(&square_avg_dL_dw, dL_dw, beta2);
	movingSquareAverage(&square_avg_dL_db, dL_db, beta2);
	weight -= (eta * (1 / (std::sqrt(square_avg_dL_dw + EPSILON))) * dL_dw);
	bias -= (eta * (1 / (std::sqrt(square_avg_dL_db + EPSILON))) * dL_db);
}

void Optimizer::adam(float64_t &weight, float64_t &bias, const float64_t dL_dw, const float64_t dL_db) {
	movingAverage(&avg_dL_dw, dL_dw, beta1);
	movingAverage(&avg_dL_db, dL_db, beta1);
	movingSquareAverage(&square_avg_dL_dw, dL_dw, beta2);
	movingSquareAverage(&square_avg_dL_db, dL_db, beta2);
	weight -= (eta * (avg_dL_dw / (std::sqrt(square_avg_dL_dw + EPSILON))));
	bias -= (eta * (avg_dL_db / (std::sqrt(square_avg_dL_db + EPSILON))));
}

Optimizer_Method_T Optimizer::getMethod(void) {
	return method;
}