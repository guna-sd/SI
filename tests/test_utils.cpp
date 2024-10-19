#include <iostream>
#include <vector>
#include <string>
#include "/home/guna/Projects/Open/SI/src/utils/command.h"
#include "/home/guna/Projects/Open/SI/src/utils/logger.h"

using namespace std;

void logMessages(Logger& logger, int Id) {
    logger.log(INFO, "Logger " + to_string(Id) + " started.", true);
    logger.log(DEBUG, "Debug message from Logger " + to_string(Id), true);
    logger.log(SUCCESS, "Success message from Logger " + to_string(Id), true);
    logger.log(WARNING, "Warning from Logger " + to_string(Id), true);
    logger.log(ERROR, "Error occurred in Logger " + to_string(Id), true);
    logger.log(CRITICAL, "Critical issue in Logger " + to_string(Id), true);
    logger.log(INFO, "Logger " + to_string(Id) + " finished.\n", true);
}


void myBuiltinCommand(const string& command, const vector<string>& args) {
    cout << "Executing built-in command: " << command << endl;
    cout << "Arguments: " << endl;
    for (const auto& arg : args) {
        cout << arg << endl;
    }
}

void test_logger()
{   
    Logger logger("test_logger.log");
    logMessages(logger, 1);
    logMessages(logger, 2);
}

void test_command()
{
    vector<string> lsArgs = {"-A"};
    Command lsCommand("ls", lsArgs);
    lsCommand.execute();

    vector<string> echoArgs = {"Hello"};
    Command echoCommand("echo", echoArgs);
    echoCommand.execute();

    vector<string> builtinArgs = {"arg1", "arg2"};
    Command builtinCommand("builtin", builtinArgs, myBuiltinCommand);
    builtinCommand.execute();
}

int main()
{
    test_logger();
    test_command();
    return 0;
}