#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "datatypes.h"
#include "optimizer_method_t.h"

#define EPSILON (10E-10)

class Optimizer {
private:
	float64_t dL_dw_velocity;
	float64_t dL_db_velocity;
	float64_t avg_dL_dw;
	float64_t avg_dL_db;
	float64_t square_avg_dL_dw;
	float64_t square_avg_dL_db;
	 
	Optimizer_Method_T method;

	float64_t eta;
	float64_t gamma;
	float64_t beta1;
	float64_t beta2;
public:
	Optimizer();

	void init(const Optimizer_Method_T in_method, 
		      const float64_t in_eta, 
		      const float64_t in_gamma, 
		      const float64_t in_beta1, 
		      const float64_t in_beta2);

	void reset();

	void normal(float64_t &weight, float64_t &bias, const float64_t dL_dw, const float64_t dL_db);

	void momentum(float64_t &weight, float64_t &bias, const float64_t dL_dw, const float64_t dL_db);

	void rms(float64_t &weight, float64_t &bias, const float64_t dL_dw, const float64_t dL_db);

	void adam(float64_t &weight, float64_t &bias, const float64_t dL_dw, const float64_t dL_db);

	Optimizer_Method_T getMethod(void);
};

#endif
