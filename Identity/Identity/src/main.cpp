#include <iostream>
#include <vector> // For usage of vector<>
#include <chrono>

#include "datatypes.h"
#include "parameters.h"

#include "batch_method_t.h"

#include "model.h"
#include "random_engine.h"

Model model;
RandomEngine random_engine;

int main(int argc, char* argv[]) {
	std::vector<float64_t> inp_vec(INPUT_DATA_SIZE);
	std::vector<float64_t> out_vec(INPUT_DATA_SIZE);
	std::vector<float64_t> predout_vec(INPUT_DATA_SIZE);

	random_engine.generate(inp_vec, INPUT_VEC_LOWER_BOUND, INPUT_VEC_UPPER_BOUND);

	/* Identity Function, so output is same as input */
	out_vec = inp_vec;
	
	model.getLogger()->openFile("C:\\Project\\Cpp\\Machine_Learning\\Deep_Learning\\Identity\\file.txt");
	model.getLogger()->logHeader("Epochs,Batch,Loss,Batch_dldw,Batch_dldb,Weight,Bias");

	model.getOptimizer()->init(OPTIMIZER_RMS, LEARNING_RATE, GAMMA, BETA1, BETA2);

	model.init(EPOCHS, BATCH_METHOD_BATCH, TRUE);
	auto start = std::chrono::high_resolution_clock::now();
	model.run(inp_vec, out_vec);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
	model.getLogger()->closeFile();

	std::getchar();

	return 0;
}