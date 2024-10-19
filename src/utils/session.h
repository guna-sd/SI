#ifndef SESSION_H
#define SESSION_H

#include "command.h"
#include <vector>
#include <optional>
#include <iostream>
#include <ctime>
#include <map>

using namespace std;

enum SessionType { Auto, Interactive, Unknown };

class Session
{

private:
    int userId;
    string username;
    bool active;
    SessionType type;
    vector<Command> commands;
    time_t start_time;
    time_t last_active_time;
    time_t expirationTime;
    bool is_suspended;
    static const int TIMEOUT_PERIOD = 600;
    void updateLastActiveTime()
    {
        last_active_time = time(0);
    }

public:
    Session(int userId, const std::string& username, SessionType type = SessionType.Auto, const vector<Command>& commands)
        : userId(userId), username(username), active(true), type(type), commands(commands),
          start_time(time(0)), last_active_time(start_time),
          expirationTime(start_time + 1800),
          is_suspended(false) {}

    int getUserId() const { return userId; }
    const std::string& getUsername() const { return username; }
    bool isActive() const { return active; }
    void endSession() { active = false; }
    bool isSuspended() const { return is_suspended; }
    void suspendSession() { is_suspended = true; }
    void resumeSession() { is_suspended = false; updateLastActiveTime(); }
    void clearCommandHistory() { commands.clear(); }
    void updateExpirationTime(time_t duration) { if(active){expirationTime = start_time + duration; }}
    bool isExpired() const { return time(0) > expirationTime; }
    void checkForTimeout() {if (isActive() && (time(0) - last_active_time > TIMEOUT_PERIOD)) {if (type == SessionType.Interactive){suspendSession();}}}
    SessionType getType() const { return type; }
    std::string toString() const
    {
        ostringstream ss;
        ss << "User ID: " << userId << ", Username: " << username << ", Active: " << (active? "Yes" : "No")
           << ", Type: " << (type == SessionType::Auto? "Auto" : (type == SessionType::Interactive? "Interactive" : "Unknown"));
        return ss.str();
    }

    void addCommand(const Command& cmd) { commands.push_back(cmd); updateLastActiveTime();}
    const vector<Command>& getCommands() const { return commands; }
    void printCommandHistory() const
    {
        cout << "\nCommand history for user: "<< username << "\n" << endl;
        for (const auto& cmd : commands) {
            cout << cmd.toString() << endl;
        }
    }
    time_t getLastActiveTime() const { return last_active_time; }
    void printSessionInfo() const
    {
        cout << "User ID: " << userId << endl;
        cout << "Username: " << username << endl;
        cout << "Active: " << (active? "Yes" : "No") << endl;
        cout << "Start Time: " << ctime(&start_time);
        cout << "Last Active Time: " << ctime(&last_active_time);
        cout << "Expiration Time: " << ctime(&expirationTime);
        cout << "Suspended: " << (is_suspended? "Yes" : "No") << endl;
        cout << "Command History: " << commands.size() << " commands" << endl;
        cout << endl;
    }
};
#endif