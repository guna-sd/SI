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
#include <cblas.h>

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

typedef struct {
    int dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int hiddlen_dim;
    int head_dim;
    int vocab_size;
    int max_seq_len;
    float eps;
} Config;

typedef struct {
    float *embeddings;
    float *attn_norm;
    float *wqkv;
    float *wo;
    float *post_attn_norm;
    float *w1;
    float *w2;
    float *w3;
    float *layer_norm;
    float *wcls;
} Weights;

typedef struct {
    float *x;
    float *x2;
    float *xt;
    float *h;
    float *h2;
    float *qkv;
    float *attn;
    float *logits;
    float *key_cache;
    float *value_cache;
} Runstate;

typedef struct {
    Config config;
    Weights weights;
    RunState runstate;
    int fd;
    float *data;
    ssize_t file_size;
} Transformer;

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

long time_in_ms();
void build_tokenizer(Tokenizer *tokenizer, char *tokenizer_path)
void free_tokenizer(Tokenizer *tokenizer)
void encode(Tokenizer* t, char *text, bool bos, bool eos, int *tokens, int *n_tokens)
char *decode(Tokenizer *tokenizer, int prev_token, int token)
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size)
void matmul(float *out,float *x, float *w, int n, int dim)
void softmax(float *x, int size);
void rms_norm(float *o, float *x, float *w, int dim, float eps) 
int compare_tokens(const void *a, const void *b);
int sample_argmax(float *prob, int n);
int sample_mult(float *prob, int n, float coin);
int topp(float *prob, int size, float topp, ProbIndex *probindex, float coin);
unsigned int random_u32(unsigned long long *state);
float random_f32(unsigned long long *state);
void forward(Transformer *transformer, int token, int pos)

#endif