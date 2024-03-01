#if !defined (_SHELL_H_)
# define _SHELL_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
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
#include <stdint.h>
#include <sys/mman.h>
#include <sys/file.h>


#define CMD_SUCCESS 0
#define CMD_FAILURE 1

#define BGREEN "\e[1;92m"
#define BRED "\e[1;91m"
#define BYELLOW "\e[1;93m"
#define BBLUE "\e[1;94m"
#define BMAGENTA "\e[1;95m"
#define BCYAN "\e[1;96m"
#define BWHITE "\e[1;97m"
#define RESET "\e[0m"

#define SI_VERSION "0.1"
#define INPUT_SIZE 1024
static char _path[1024];

// shell utils
static void init();
void _exec(char **args);
void _exec_bg(char **args);
void _io(char *args[], char *outputFile, int option);
void _pipe(char *args[]);
int _cmdh(char *args[]);
static void signalHandler_child();
static void signalHandler_interrupt();

// cmd utils
int _cd(char *argv[]);
void _help(void);
void _history(void);
void _perror(char *ErrorMessage);
void initial_screen(void);
void prompt(void);
char* _input(void);
void _store(char *args[]);
void _clr_history(void);
void _out();

// llm utils

typedef struct llm_config {
    int dim;
    int hidden_dim;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int max_seq_len;
    int vocab_size;
} Config;

typedef struct llm_weights {
    float *token_embedding_table;
    float *rms_att_weights;
    float *rms_ffn_weights;
    float *wq;
    float *wk;
    float *wv;
    float *wo;
    float *w1;
    float *w2;
    float *w3;
    float *rms_final_weights;
    float *wcls;
} Weights;

typedef struct llm_runstate {
    float *x;
    float *xb;
    float *xb2;
    float *hb;
    float *hb2;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *logits;
    float *key_cache;
    float *value_cache;
} Runstate;

typedef struct llm_transformer 
{
    Config config;
    Weights weights;
    Runstate runstate;
    int fd;
    float *data;
    size_t file_size;
} Transformer;

typedef struct llm_tokens
{
    char *token;
    int token_id;
} Tokens;

typedef struct llm_tokenizer
{
    char **vocab;
    float *vocab_scores;
    Tokens *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

typedef struct llm_P
{
    float prob;
    int index;
} ProbIndex;

typedef struct llm_sampler
{
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

long time_in_ms();
void bprintf(char *piece);
void matmul(float *out, float *x, float *y, int n, int d);
void softmax(float* x, int size);
void rmsnorm(float* o, float* x, float* weight, int size);
int compare_tokens(const void *a, const void *b);
int compare(const void *a, const void *b);
int sample_argmax(float *prob, int n);
int sample_mult(float *prob, int n, float coin);
int topp(float *prob, int size, float topp, ProbIndex *probindex, float coin);
unsigned int random_u32(unsigned long long *state);
float random_f32(unsigned long long *state);
void *read_file(char *filename);

#endif