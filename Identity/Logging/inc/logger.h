#ifndef LOGGER_H
#define LOGGER_H

#include <fstream>
#include <string>

#include "datatypes.h"

class Logger {
private:
	std::ofstream file;
	std::string file_path;

public:
	Logger();

	void setFilePath(const std::string &in_file_path);

	void openFile(const std::string &in_file_path);

	void logHeader(const std::string &header);

	void logData(const uint32_t epoch,
		const uint32_t batch,
		const float64_t loss,
		const float64_t dL_dw,
		const float64_t dL_db,
		const float64_t weight,
		const float64_t bias);

	void closeFile(void);
};

#endif