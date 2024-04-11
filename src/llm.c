#include "../include/shell.h"


int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void build_tokenizer(Tokenizer *tokenizer, char *tokenizer_path)
{
    for (int i =0; i < 256 ; i++) {
        tokenizer->byte_pieces[i *2] = (unsigned char)i;
        tokenizer->byte_pieces[i *2 + 1] = '\0';
        }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file)
    {
        fprintf(stderr, "Could not open file %s\n", tokenizer_path);
        exit(1);
    }
    if(fread(&tokenizer->max_token_length, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "Could not read max token length %s\n", tokenizer_path);
        exit(1);
    }
    if(fread(&tokenizer->vocab_size, sizeof(int), 1, file)!= 1)
    {
        fprintf(stderr, "Could not read vocab size %s\n", tokenizer_path);
        exit(1);
    }
    int vocab_size = tokenizer->vocab_size;
    tokenizer->vocab = (char**)malloc(vocab_size * sizeof(char*));
    tokenizer->vocab_scores = (float*)malloc(vocab_size * sizeof(float));

    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(tokenizer->vocab_scores + i, sizeof(float), 1, file) != 1)
        { 
            fprintf(stderr, "failed read\n"); 
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) 
        { 
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE); 
        }
        tokenizer->vocab[i] = (char *)malloc(len + 1);
        if (fread(tokenizer->vocab[i], len, 1, file) != 1) 
        {
             fprintf(stderr, "failed read\n"); 
             exit(EXIT_FAILURE); 
        }
        tokenizer->vocab[i][len] = '\0';
    }
    if ((tokenizer->sorted_vocab = malloc(vocab_size * sizeof(TokenIndex))))
    {
        for (int i =0; i < tokenizer->vocab_size; i++)
        {
            tokenizer->sorted_vocab[i].id = i;
            tokenizer->sorted_vocab[i].str = tokenizer->vocab[i];
        }
        qsort(tokenizer->sorted_vocab, tokenizer->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    fclose(file);
    return;
}

void free_tokenizer(Tokenizer *tokenizer)
{
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->vocab[i]);
    }
    free(tokenizer->vocab);
    free(tokenizer->vocab_scores);
    free(tokenizer->sorted_vocab);
}

void encode(Tokenizer* t, char *text, bool bos, bool eos, int *tokens, int *n_tokens) {

    if (text == NULL) { 
        fprintf(stderr, "cannot encode NULL text\n"); 
        exit(EXIT_FAILURE); 
        }

    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    *n_tokens = 0;

    if (bos == true) {
        tokens[(*n_tokens)++] = 2;
    }

    for (char *c = text; *c != '\0'; c++) {

        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;
        }

        tokens[best_idx] = best_id;
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    if (eos == true) {
        tokens[(*n_tokens)++] = 1;
    }
    free(str_buffer);
}


char *decode(Tokenizer *tokenizer, int prev_token, int token)
{
    char *string = tokenizer->vocab[token];

    if (prev_token == 1 && string[0] == ' ')
    {
        string++;
    }
    unsigned char byte_val;
    if (sscanf(string, "<0x%02hhX>", &byte_val) == 1)
    {
        string = (char *)tokenizer->byte_pieces + byte_val * 2;
    }
    return string;
}

void matmul(float *out,float *x, float *w, int n, int dim)
{
    int i,j;
    #pragma omp parallel for private(i, j)
    for (i = 0; i < dim; i++)
    {
        float val = 0.0f;
        for (j = 0; j < n; j++)
        {
            val += w[i *n + j] * x[j];
        }
        out[i] = val;
    }
}

void rms_norm(float *o, float *x, float *w, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum_sq += x[i] * x[i];
    }
    sum_sq /= dim
    float inv_sqrt = 1.0f / sqrtf(sum_sq + eps);

    for (int i = 0; i < dim; i++) {
        o[i] = x[i] * inv_sqrt * (w[j] + 1.0f);
    }
}


void softmax(float *x, int size)
{
    float max = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max)
        {
            max = x[i];
        }
    }
    float sum = 0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

float* forward(Transformer *transformer, int token, int pos)
{
    Config *params = &transformer->config;
    Weights *weights = &transformer->weights;
    RunState *runstate = &transformer->runstate;
    float *embeddings = weights->embeddings;

    float *x = runstate->x;

    int dim = params->dim;
    int kv_dim = (params->dim * params->n_kv_heads) / params->n_heads;
    int head_dim = params->head_dim;
    int hidden_dim = params->hidden_dim;
    int num_heads = params->n_heads;
    int head_size = dim / num_heads;
    int kv_mul = num_heads / params->n_kv_heads;
    float eps = params->eps

    float *embeddings = weights->embeddings + token * dim;
    memcpy(x, embeddings, dim * sizeof(*x));

    for (unsigned long long layer = 0; layer < params->n_layers;layer++)
    {
        rms_norm(runstate->x2, x, weights->attn_norm + layer * dim, dim, eps);

        int layer_offset = layer * params->max_seq_len * kv_dim;
        runstate->k = runstate->key_cache + layer_offset + pos * kv_dim;
        runstate->v = runstate->value_cache + layer_offset + pos * kv_dim;

        matmul(runstate->q, runstate->xb, runstate->wq + layer * dim * dim, dim, dim);
        matmul(runstate->k, runstate->xb, runstate->wk + layer * dim * kv_dim, dim, kv_dim);
        matmul(runstate->v, runstate->xb, runstate->wv + layer * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i+2)
        {
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq
            float fcs = cosf(val);
            float fsi = sinf(val);
            int rotational = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotational; v++)
            {
                float vect = v == 0 ? state->q : state->k;
                float v0 = vect[i];
                float v1 = vect[i + 1];
                vect[i] = v0 * fcs - v1 * fsi;
                vect[i + 1] = v0 * fsi + v1 * fcs;
            }
        }

        int h;
        #pragma omp parallel for private(h)
        for (h=o; h < num_heads; h++)
        {
            float *q = state->q + h * head_size;
            float *att = state->att + h * params->max_seq_len;
            for (int t=0; t <= pos; t++)
            {
                float *k = state->key_cache + layer_offset + t * kv_dim + (h/kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++)
                {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);

            float *xb = state->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t=0; t <= pos; t++)
            {
                float *v = state->value_cache + layer_offset + t * kv_dim + (h/kv_mul) * head_size;
                for (int i = 0; i < head_size; i++)
                {
                    xb[i] += att[t] * v[i];
                }
            }
        }
        matmul(state->xb2, state->xb, weights->wo + layer * dim * dim, dim, dim);

        for (int i = 0; i < dim; i++)
        {
            x[i] += state->xb2[i];
        }

        rms_norm(state->xb, x, weights->post_attn_norm + layer * dim, dim, eps)

        matmul(state->hb, state->xb, weights->w1 + layer * dim * hidden_dim, dim, hidden_dim);
        matmul(state->hb, state->xb, weights->w3 + layer * dim * hidden_dim, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; i++)
        {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= state->hb2[i];
            state->hb[i] = val;
        }

        matmul(state->xb, state->hb weights->w2 + layer * dim * hidden_dim, hidden_dim, dim);

        for (int i = 0; i < dim; i++)
        {
            x[i] += state->xb[i];
        }
    }
    rms_norm(x,x,weights->layer_norm,dim,eps);

    matmul(state->logits, x, weights->wcls, params->dim, params->vocab_size);
    return state->logits;
}

