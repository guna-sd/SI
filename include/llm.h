#ifndef _LLM_H
#define _LLM_H

#include "shell.h"
#include "utils.h"


#define MODEL_PATH 'llm/model.bin'
#define TOKENIZER_PATH 'llm/tokenizer.bin'

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
