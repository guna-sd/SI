#include "../include/shell.h"
#include "../include/util.h"
#include "../include/llm.h"
void init()
{
    _terminal = STDIN_FILENO;
    shell_is_interactive = isatty(_terminal);
    SHELL_PID = getpid();
    if (shell_is_interactive)
    {
        while (tcgetpgrp(_terminal) != (SHELL_PGID=getpgrp()))
        {
            kill(SHELL_PID,SIGTTIN);
        }
        signal(SIGINT, signalHandler_interrupt);
        signal(SIGCHLD, signalHandler_child);

        setpgid(SHELL_PID, SHELL_PID);
        SHELL_PGID = getpgrp();
        if (SHELL_PID != SHELL_PGID)
        {
            _perror("Error, the shell is not process group leader");
            exit(1);
        }
        tcsetpgrp(_terminal, SHELL_PGID);
        tcgetattr(_terminal, &_shell);
        currentDirectory = (char*) calloc(1024, sizeof(char));
        getcwd(currentDirectory, 1024);
    }
    else 
    {
        _perror("Error : Could not set SHELL as interactive...");
        exit(1);
    }
}

void _perror(char *ErrorMessage)
{
    fprintf(stderr, BRED);
    perror(ErrorMessage);
    fprintf(stderr, BWHITE);
}

int main(int argc, char *argv[], char **envp) {
    char *tokens[512];
    int ntok;
    pid = -10;
    init();
    initial_screen();

    _environment = envp;

    setenv("shell", getcwd(currentDirectory, 1024), 1);

    while (1) {
        prompt();
        char *input = _input();
        if ((tokens[0] = strtok(input, " \n\t")) == NULL) {
            continue;
        }
        ntok = 1;
        while ((tokens[ntok] = strtok(NULL, " \n\t")) != NULL) ntok++;
        CMD_RUNTIME(tokens);
    }
    return 0;
}
