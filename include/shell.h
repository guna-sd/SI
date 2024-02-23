#if !defined (_SHELL_H_)
#define _SHELL_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
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
#define CMD_ERROR -1

#define _EXEC    0 
#define OR_EXEC  1  // ---> EXECUTION OF COMMAND USING LOGICAL OR '||'
#define AND_EXEC 2  // ---> EXECUTION OF COMMAND USING LOGICAL AND '&&'
#define SEQ_EXEC 3  // ---> EXECUTION OF COMMAND FOLLOWED BY SEMICOLON ';'
#define REDIRECT_EXEC 4 // ---> EXECUTION OF COMMAND FOLLOWED BY '>'
#define PIPE_EXEC 5  // ---> EXECUTION OF COMMAND FOLLOWED BY '|'

#define BGREEN "\e[1;92m"
#define BRED "\e[1;91m"
#define BYELLOW "\e[1;93m"
#define BBLUE "\e[1;94m"
#define BMAGENTA "\e[1;95m"
#define BCYAN "\e[1;96m"
#define BWHITE "\e[1;97m"
#define RESET "\e[0m"

#define SI_VERSION "0.1"
#define HIST_FILE	".si_history"
#define HIST_MAX_Lines 1024
#define MAX_ALIAS '100'
#define MAX_CMD_SIZE 1024


typedef struct {
    char *cmd;
    char *alias;
    char *path;
} Alias;

typedef struct process
{
  struct process *next;
  char **argv;
  pid_t pid;
  char completed;
  char stopped;
  int status;
} process;


// shell utils
void init();
int _fork(char *argv[]);
int _exec(char *argv[]);
int _exec_or(char *argv[]);
int _exec_and(char *argv[]);
int _exec_seq(char *argv[]);
int _exec_redirect(char *args, int mode, char *filename);
int _exec_pipe(char *args);
void _perror(char *ErrorMessage);
void signalHandler_child();
void signalHandler_interrupt();

// cmd utils
int _cd(char *argv[]);
int _alias(char *argv[]);
int _version();
int _help();
int _history();

// shell utils
void initial_screen();
int _add_alias(char *args, char *alias);
int _remove_alias(char *alias);
void prompt();
char* _input();

// llm utils
long time_in_ms();
void safe_printf(char *piece);
void matmul(float* xout, float* x, float* w, int n, int d);
void softmax(float* x, int size);
void rmsnorm(float* o, float* x, float* weight, int size);
int compare_tokens(const void *a, const void *b);
int compare(const void *a, const void *b);
int sample_argmax(float *prob, int n);
int sample_mult(float *prob, int n, float coin);
unsigned int random_u32(unsigned long long *state);
float random_f32(unsigned long long *state);
void *read_file(char *filename);


#endif
