#include "../include/shell.h"
#include "../include/util.h"
#include "../include/llm.h"

int init()
{
    shell_is_interactive = isatty(STDIN_FILENO);
    if (shell_is_interactive)
    {
        while (tcgetpgrp(STDIN_FILENO) != (SHELL_PGID=getpgrp()))
        {
            kill(SHELL_PGID,SIGTTIN);
        }
    }
    return 0;
}

int main() {
    char * tokens[256];
    initial_screen();
    while(1)
    {
    char* msg = _input();
    
    }
    return 0;
}
