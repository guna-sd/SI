#ifndef _SHELL_H_
#define _SHELL_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <termios.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <limits.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include <sys/mman.h>

#define CMD_SUCCESS 0
#define CMD_FAILURE 1
#define CMD_ERROR -1
#define CMD_MAX_SIZE 1024

#define _EXEC    0 
#define OR_EXEC  1  // ---> EXECUTION OF COMMAND USING LOGICAL OR '||'
#define AND_EXEC 2  // ---> EXECUTION OF COMMAND USING LOGICAL AND '&&'
#define SEQ_EXEC 3  // ---> EXECUTION OF COMMAND FOLLOWED BY SEMICOLON ';'
#define REDIRECT_EXEC 4 // ---> EXECUTION OF COMMAND FOLLOWED BY '>'
#define PIPE_EXEC 5  // ---> EXECUTION OF COMMAND FOLLOWED BY '|'

static pid_t SHELL_PGID;
static int shell_is_interactive;
struct termios shell_tmodes;


// shell utils
int init();
int _fork(char *args);
int _exec(char *args);
int _exec_or(char *args);
int _exec_and(char *args);
int _exec_seq(char *args);

// cmd utils
int _cd(char *argv[]);
int _alias(char *argv[]);
int _echo(char *argv[]);
int _clear();
int _version();
int _quit();
int _help();
int _history();



#endif
