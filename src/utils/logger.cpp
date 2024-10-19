#include <logging.h>

void Logger::log(const std::string& message, LogLevel level) 
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    
    std::string levelStr;
    switch (level) {
        case INFO: levelStr = "INFO"; break;
        case WARNING: levelStr = "WARNING"; break;
        case ERROR: levelStr = "ERROR"; break;
    }

    std::cout << "[" << std::ctime(&now_time) << "] [" << levelStr << "] " << message << std::endl;
}
