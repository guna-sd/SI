#ifndef __UTILS_
#define __UTILS_

#include "shell.h"

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

static char inputbuffer[CMD_MAX_SIZE];

typedef struct {
    char *cmd;
    char *alias;
    char *path;
} Alias;

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

// terminal utils
void initial_screen();
void prompt();
char* _input();

// shell utils
int _add_alias(char *args, char *alias);
int _remove_alias(char *alias);


#endif