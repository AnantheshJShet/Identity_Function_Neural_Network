#include <algorithm>
#include <assert.h>

#include "model.h"

#include "parameters.h"

Model::Model() {

}

void Model::init(const uint32_t in_epochs,
	             const Batch_Method_T batch_method,
	             const boolean_t in_schuffle_input) {
	random_engine.generate(weight, PARAMETERS_LOWER_BOUND, PARAMETERS_UPPER_BOUND);
	random_engine.generate(bias, PARAMETERS_LOWER_BOUND, PARAMETERS_UPPER_BOUND);
	total_epochs = in_epochs;
	epoch = 0;

	if (BATCH_METHOD_STOCHASTIC == batch_method) {
		batch_size = 1;
	}
	else if (BATCH_METHOD_MINI_BATCH == batch_method) {
		batch_size = MINI_BATCH_SIZE;
	}
	else if (BATCH_METHOD_BATCH == batch_method) {
		batch_size = INPUT_DATA_SIZE;
	}
	schuffle_input = in_schuffle_input;
	pred_out = 0;
	loss = 0;
	dL_dw = 0;
	dL_db = 0;
}

void Model::setInOut(const float64_t in_inp, const float64_t in_out) {
	inp = in_inp;
	out = in_out;
}

void Model::forwardProp(void) {
	/* yhat = w*x + b */
	pred_out = (inp * weight) + bias;
}

void Model::findLoss(void) {
	/* L = 0.5*(yhat - y)^2 */
	loss = 0.5*std::pow((pred_out - out), 2);
}

void Model::backwardProp() {
	/* 
	* dL_dw = dL_dyhat*dyhat_dw
	* dL_db = dL_dyhat*dyhat_db
	* dL_dyhat = (yhat-y)
	* dyhat_dw = x
	* dyhat_db = 1 
	*/
	dL_dw = (pred_out - out) * inp;
	dL_db = (pred_out - out);

	assert(out == inp);

	sample_num++;

	batch_loss += loss;
	batch_dL_dw += dL_dw;
	batch_dL_db += dL_db;
	if (0 == (sample_num % batch_size)) {
		batch_num++;
		batch_loss /= batch_size;
		batch_dL_dw /= batch_size;
		batch_dL_db /= batch_size;
		updateWeights();
		logger.logData(epoch, batch_num, batch_loss, batch_dL_dw, batch_dL_db, weight, bias);
		batch_loss = 0;
		batch_dL_dw = 0;
		batch_dL_db = 0;
	}
}

void Model::checkWeights(void) {
	float64_t weight_diff = std::abs(1 - weight);
	float64_t bias_diff = std::abs(0 - bias);
	if ((weight_diff < DIFF_THRESHOLD) && (bias_diff < DIFF_THRESHOLD)) {
		status = MODEL_STATUS_TRAINING_STOP;
	}

}

void Model::updateWeights() {
	switch (optimizer.getMethod()) {
	case OPTIMIZER_GD:
		optimizer.normal(weight, bias, batch_dL_dw, batch_dL_db);
		break;
	case OPTIMIZER_GD_WITH_MOMENTUM:
		optimizer.momentum(weight, bias, batch_dL_dw, batch_dL_db);
		break;
	case OPTIMIZER_RMS:
		optimizer.rms(weight, bias, batch_dL_dw, batch_dL_db);
		break;
	case OPTIMIZER_ADAM:
		optimizer.adam(weight, bias, batch_dL_dw, batch_dL_db);
		break;
	default:
		break;
	}
}

void Model::run(std::vector<float64_t> &inp_vec, std::vector<float64_t> &out_vec) {
	status = MODEL_STATUS_TRAINING_START;
	while ((epoch < total_epochs) && (MODEL_STATUS_TRAINING_STOP != status)) {
		epoch++;
		if (TRUE == schuffle_input) {
			std::random_shuffle(inp_vec.begin(), inp_vec.end());
			out_vec = inp_vec;
		}

		for (uint32_t idx = 0; idx < inp_vec.size(); ++idx) {
			setInOut(inp_vec[idx], out_vec[idx]);
			forwardProp();
			findLoss();
			backwardProp();
			checkWeights();
			if (MODEL_STATUS_TRAINING_STOP == status) {
				break;
			}
		}

		batch_num = 0;

		/* Reset all the internal variables of the optimizer after every epoch */
		optimizer.reset();
	}
	status = MODEL_STATUS_TRAINING_STOP;
}

float64_t Model::getWeight(void) {
	return weight;
}

float64_t Model::getBias(void) {
	return bias;
}

uint32_t Model::getTotalEpochs(void) {
	return total_epochs;
}

uint32_t Model::getEpochNum(void) {
	return epoch;
}

Optimizer* Model::getOptimizer(void) {
	return &optimizer;
}

Logger* Model::getLogger(void) {
	return &logger;
}