#ifndef LOGGER_H
#define LOGGER_H

#include <string>

class Logger {
public:
    enum LogLevel {
        INFO,
        WARNING,
        ERROR,
    };

    Logger();
    void log(const std::string& message, LogLevel level = LogLevel.INFO);
};

#endif