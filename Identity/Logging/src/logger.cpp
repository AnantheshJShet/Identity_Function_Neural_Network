
#include "logger.h"

Logger::Logger() {
	file_path = "";
}

void Logger::setFilePath(const std::string &in_file_path) {
	file_path = in_file_path;
}

void Logger::openFile(const std::string &in_file_path) {
	file_path = in_file_path;
	if (FALSE == file_path.empty()) {
		file.open(file_path);
	}
}

void Logger::logHeader(const std::string &header) {
	file << header << std::endl;
}

void Logger::logData(const uint32_t epoch,
	const uint32_t batch,
	const float64_t loss,
	const float64_t dL_dw,
	const float64_t dL_db,
	const float64_t weight,
	const float64_t bias){
	file << epoch << "," << batch << "," << loss << "," << dL_dw << "," << dL_db << "," << weight << "," << bias << std::endl;
}

void Logger::closeFile() {
	file.close();
}