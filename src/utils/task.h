#ifndef TASK_H
#define TASK_H

#include <string>
#include <vector>
#include <ctime>
#include "command.h"

class Task 
{

public:
    Task(int id, const std::string& name, const std::string& recurrence, const std::vector<Command>& cmds)
    : taskID(id), taskName(name), recurrence(recurrence), nextRun(0), lastRun(0), associatedCommands(cmds), 
      isActive(true), taskStatus("pending"), owner(""), priority(0) {}
    
    

private:
    int taskID;
    std::string taskName;
    std::string recurrence;
    std::time_t nextRun;
    std::time_t lastRun;
    std::vector<Command> associatedCommands;
    bool isActive;                       // Indicates if the task is active
    std::string taskStatus;              // Current status of the task (e.g., "pending", "completed", "failed")
    std::string owner;                   // User or system that owns the task
    int priority;
}


#endif