#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

enum LogLevel { DEBUG, INFO, WARNING, ERROR, CRITICAL, SUCCESS };

class Logger {
public:
    Logger(const string& filename)
    {
        logFile.open(filename, ios::app);
        if (!logFile.is_open()) {
            cerr << "Error opening log file." << endl;
        }
    }
    ~Logger() 
    { 
        if (logFile.is_open()) {
            logFile.close(); 
        }
    }


    void log(LogLevel level, const string& message, bool printlog)
    {
        time_t now = time(0);
        tm* timeinfo = localtime(&now);
        char timestamp[20];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", timeinfo);
        
        ostringstream logEntry;
        logEntry << "[" << timestamp << "] "
                 << plainLevelToString(level) << ": " << message << endl;

        if (printlog)
        {
        cout << "\033[32m[" << timestamp << "]\033[0m "
             << coloredLevelToString(level) << ": " << message << "\033[0m"
             << endl;
        }

        if (logFile.is_open()) 
        {
            logFile << logEntry.str();
            logFile.flush();
        }
    }
private:
    ofstream logFile;
    string plainLevelToString(LogLevel level)
    {
        switch (level) {
            case DEBUG: return "DEBUG";
            case INFO: return "INFO";
            case WARNING: return "WARNING";
            case ERROR: return "ERROR";
            case CRITICAL: return "CRITICAL";
            case SUCCESS: return "SUCCESS";
            default: return "UNKNOWN";
        }
    }

    string coloredLevelToString(LogLevel level)
    {
        switch (level) {
            case DEBUG: return "\e[1;94mDEBUG";
            case INFO: return "\033[1;97mINFO";
            case WARNING: return "\033[1;93mWARNING";
            case ERROR: return "\033[1;95mERROR";
            case CRITICAL: return "\033[1;91mCRITICAL";
            case SUCCESS: return "\033[1;92mSUCCESS";
            default: return "\033[36mUNKNOWN";
        }
    }

};

void logmessage(Logger* logger = nullptr, LogLevel level, const string& message, bool printlog = false)
{
    if (logger != nullptr)
    {
        logger->log(level, message, printlog);
        return;
    }
    cout << "logger Failed: " << level << message << endl;
    return;
}