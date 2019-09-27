#ifndef MODEL_H
#define MODEL_H

#include <vector>

#include "datatypes.h"
#include "batch_method_t.h"
#include "model_status_t.h"
#include "optimizer.h"
#include "logger.h"
#include "random_engine.h"

class Model {
private:
	/* Single sample variables */
	float64_t inp;
	float64_t out;
	float64_t weight;
	float64_t bias;
	float64_t pred_out;
	float64_t loss;
	float64_t dL_dw;
	float64_t dL_db;

	/* Single batch variables */
	float64_t batch_loss;
	float64_t batch_dL_dw;
	float64_t batch_dL_db;

	uint32_t sample_num;
	uint32_t batch_num;
	uint32_t batch_size;
	uint32_t total_epochs;
	uint32_t epoch;

	RandomEngine random_engine;
	Optimizer optimizer;
	Logger logger;

	Model_Status_T status;
	boolean_t schuffle_input;
public:
	Model();

	void init(const uint32_t in_epochs, 
		      const Batch_Method_T batch_method,
		      const boolean_t in_schuffle_input);

	void setInOut(const float64_t in_inp, const float64_t in_out);

	void forwardProp(void);

	void findLoss(void);

	void backwardProp(void);

	void updateWeights(void);

	void checkWeights(void);

	void run(std::vector<float64_t> &inp_vec, std::vector<float64_t> &out_vec);

	float64_t getWeight(void);

	float64_t getBias(void);

	uint32_t getTotalEpochs(void);

	uint32_t getEpochNum(void);

	Optimizer* getOptimizer(void);

	Logger* getLogger(void);
};

#endif
