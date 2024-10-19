#ifndef SESSION_MANAGER_H
#define SESSION_MANAGER_H

#include <map>
#include <memory>
#include <iostream>
#include "session.h"

using namespace std;

class SessionManager
{
private:
    static const int MaxActiveSessions = 10;
    int sessionCount = 0;
    int suspendedSessionsCount = 0;
    int cleanupInterval = 300;
    map<string, int> userSessionMap;
    map<int, unique_ptr<Session>> sessions;

public:
    void create(int userId, const string& username, SessionType type = SessionType.Interactive)
    {
        if (sessions.find(userId) != sessions.end())
        {
            cerr << "Session for user " << userId << " already exists." << endl;
            return;
        }
        sessions[userId] = make_unique<Session>(userId, username, type);
        cout << "Created new session for user " << username << endl;
    }

    void addCommand(int userId, const Command& cmd)
    {
        if (sessions.find(userId) == sessions.end())
        {
            cerr << "No session found for user " << userId << "." << endl;
            return;
        }
        sessions[userId]->addCommand(cmd);
    }

    void endSession(int userId)
    {
        if (sessions.find(userId) == sessions.end())
        {
            cerr << "No session found for user " << userId << "." << endl;
            return;
        }
        sessions[userId]->endSession();
        cout << "Session for user " << sessions[userId]->getUsername() << " ended." << endl;
    }

    void suspendSession(int userId)
    {
        if (sessions.find(userId) == sessions.end())
        {
            cerr << "No session found for user " << userId << "." << endl;
            return;
        }
        sessions[userId]->suspendSession();
        cout << "Session for user " << sessions[userId]->getUsername() << " suspended." << endl;
    }

    void resumeSession(int userId)
    {
        if (sessions.find(userId) == sessions.end())
        {
            cerr << "No session found for user " << userId << "." << endl;
            return;
        }
        sessions[userId]->resumeSession();
        cout << "Session for user " << sessions[userId]->getUsername() << " resumed." << endl;
    }

    void printSessionInfo(int userId) const
    {
        if (sessions.find(userId) == sessions.end())
        {
            cerr << "No session found for user " << userId << "." << endl;
            return;
        }
        sessions[userId]->printSessionInfo();
    }

    void printAllSessions() const {
        if (sessions.empty()) {
            cout << "No active sessions." << endl;
            return;
        }
        for (const auto& pair : sessions) {
            pair.second->printSessionInfo();
        }
    }

    void clearSessions() {
        sessions.clear();
        cout << "All sessions cleared." << endl;
    }
};
#endif