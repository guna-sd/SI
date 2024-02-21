#include "../include/shell.h"


void initial_screen()
{
    printf(BRED);
    printf("  ____   _            _  _  ____          _   \n");
    printf(" / ___| | |__    ___ | || || __ )   ___  | |_ \n");
    printf(" \\___ \\ | '_ \\  / _ \\| || ||  _ \\  / _ \\ | __|\n");
    printf("  ___) || | | ||  __/| || || |_) || (_) || |_ \n");
    printf(" |____/ |_| |_| \\___||_||_||____/  \\___/  \\__|\n");
    printf("\n"BYELLOW);
    printf("An simple shell with the integration of llm (tinyllama) in c.");
    printf("\n"RESET);
    printf("\n");
}

void prompt()
{
    char hostn[1204] = "";
    char currentDirectory[1204] = "";
	gethostname(hostn, sizeof(hostn));
    printf(BGREEN"%s@%s:"BBLUE"[%s]"BWHITE"$!> ",getenv("LOGNAME"), hostn, getcwd(currentDirectory, 1024));
    fflush(stdout);
}

char* _input()
{
    memset(inputbuffer, '\0', sizeof(inputbuffer));
    prompt();
    fgets(inputbuffer, CMD_MAX_SIZE, stdin);
    return inputbuffer;
}