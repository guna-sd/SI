#ifndef CMD_H
#define CMD_H

#include <string>
#include <vector>
#include <map>
#include <ctime>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <cstring>
#include <iostream>
#include <functional>
#include <optional>

class Command
{
public:
    Command(const std::string& cmd, const std::vector<std::string>& args = {}, 
            std::function<void(const std::string&, const std::vector<std::string>&)> builtinFunc = nullptr)
        : command(cmd), arguments(args), BuiltinFunction(builtinFunc), 
          is_executed(false), is_error(false) {}

    void execute()
    {
        if (is_executed){ return; }
        is_executed = true;

        if (BuiltinFunction) 
        {
            BuiltinFunction(command, arguments);
        }
        else 
        {
            pid_t pid = fork();
            if (pid == -1) 
            {
                is_error = true;
                error_msg = "Fork failed: " + std::string(strerror(errno));
                return;
            }
            else if (pid == 0) 
            {
                std::vector<char*> c_args;
                c_args.push_back(const_cast<char*>(command.c_str()));
                for (const auto& arg : arguments)
                {
                    c_args.push_back(const_cast<char*>(arg.c_str()));
                }
                c_args.push_back(nullptr);
                
                if(execvp(c_args[0], c_args.data()) == -1)
                {
                    is_error = true;
                    error_msg = "Execvp failed: " + std::string(strerror(errno));
                    exit(1); 
                }
                else
                {
                    int status;
                    waitpid(pid, &status, 0);
                    if (WIFEXITED(status) && WEXITSTATUS(status) != 0)
                    {
                        is_error = true;
                        error_msg = "Command exited with status: " + std::to_string(WEXITSTATUS(status));
                    }
                }
            }
        }
    }

    std::string toString() const
    {
        std::string result = command;
        for (const auto& arg : arguments) {
            result += " " + arg;
        }
        return result;
    }
    bool hasError() const { return is_error.value_or(false); }
    std::optional<std::string> getErrorMessage() const { return error_msg; }
    std::string getCommand() const { return command; }
    std::vector<std::string> getArguments() const { return arguments; }
    bool isExecuted() const { return is_executed; }
    bool hasBuiltinFunction() const { return BuiltinFunction != nullptr; }
private:
    std::string command;
    std::vector<std::string> arguments;
    std::function<void(const std::string&, const std::vector<std::string>&)> BuiltinFunction;
    std::bool is_executed;
    std::bool is_error;
    std::optional<std::string> error_msg;

}
///home/guna/Projects/SI/src/utils/command.h
#endif