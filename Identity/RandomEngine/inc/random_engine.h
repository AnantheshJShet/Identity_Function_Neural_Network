#ifndef RANDOM_ENGINE_H
#define RANDOM_ENGINE_H

#include "datatypes.h"

class RandomEngine {
private:
public:
	RandomEngine();

	void generate(std::vector<float64_t> &vec,
				  const float64_t lower_bound,
		          const float64_t upper_bound);

	void generate(float64_t &var,
				  const float64_t lower_bound,
				  const float64_t upper_bound);
};

#endif