#include <chrono>
#include <algorithm>
#include <random>

#include "random_engine.h"

RandomEngine::RandomEngine() {

}

void RandomEngine::generate(std::vector<float64_t> &vec, 
	                        const float64_t lower_bound,
	                        const float64_t upper_bound) {
	auto seed = 1;// std::chrono::system_clock::now().time_since_epoch().count(); // Seed
	std::default_random_engine dre(seed); // Engine
	std::uniform_real_distribution<float64_t> dis(lower_bound, upper_bound); // Distribution
	std::generate(vec.begin(), vec.end(), [&] {return dis(dre); });
}

void RandomEngine::generate(float64_t &var,
	                        const float64_t lower_bound,
	                        const float64_t upper_bound) {
	auto seed = std::chrono::system_clock::now().time_since_epoch().count(); // Seed
	std::default_random_engine dre(seed); // Engine
	std::uniform_real_distribution<float64_t> dis(lower_bound, upper_bound); // Distribution
	var = dis(dre);
}