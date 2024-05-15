#include "../include/shell.h"


int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void tokenizer(Tokenizer *t, char *tokenizer_path)
{
    for (int i =0; i < 256 ; i++) {
        t->byte_pieces[i *2] = (unsigned char)i;
        t->byte_pieces[i *2 + 1] = '\0';
        }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file)
    {
        _perror("Could not open file Tokenizer model\n");
        exit(1);
    }
    if(fread(&t->max_token_length, sizeof(int), 1, file) != 1)
    {
        _perror("Could not read max token length from Tokenizer model\n");
        exit(1);
    }
    if(fread(&t->vocab_size, sizeof(int), 1, file)!= 1)
    {
        _perror("Could not read vocab size from Tokenizer model\n");
        exit(1);
    }
    int vocab_size = t->vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));

    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1)
        { 
            _perror("failed reading Tokenizer\n"); 
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) 
        { 
            _perror("failed reading Tokenizer\n");
            exit(EXIT_FAILURE); 
        }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) 
        {
             _perror("failed reading Tokenizer\n"); 
             exit(EXIT_FAILURE); 
        }
        t->vocab[i][len] = '\0';
    }
    if ((t->sorted_vocab = malloc(vocab_size * sizeof(TokenIndex))))
    {
        for (int i =0; i < t->vocab_size; i++)
        {
            t->sorted_vocab[i].id = i;
            t->sorted_vocab[i].str = t->vocab[i];
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    fclose(file);
    return;
}

void free_tokenizer(Tokenizer *t)
{
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

void encode(Tokenizer* t, char *text, bool bos, bool eos, int *tokens, int *n_tokens) {

    if (text == NULL) { 
        _perror("cannot encode NULL text\n"); 
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


char *decode(Tokenizer *t, int prev_token, int token)
{
    char *string = t->vocab[token];

    if (prev_token == 1 && string[0] == ' ')
    {
        string++;
    }
    unsigned char byte_val;
    if (sscanf(string, "<0x%02hhX>", &byte_val) == 1)
    {
        string = (char *)t->byte_pieces + byte_val * 2;
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
    sum_sq /= dim;
    float inv_sqrt = 1.0f / sqrtf(sum_sq + eps);

    for (int i = 0; i < dim; i++) {
        o[i] = x[i] * inv_sqrt * (w[i] + 1.0f);
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
    float sum = 0.0f;
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

float *forward(Transformer *transformer, int token, int pos)
{
    Config *params = &transformer->config;
    Weights *weights = &transformer->weights;
    Runstate *runstate = &transformer->runstate;

    float *x = runstate->x;

    int dim = params->dim;
    int kv_dim = (params->dim * params->n_kv_heads) / params->n_heads;
    int head_dim = params->head_dim;
    int hidden_dim = params->hidden_dim;
    int num_heads = params->n_heads;
    int head_size = dim / num_heads;
    int kv_mul = num_heads / params->n_kv_heads;
    float eps = params->eps;

    float *embeddings = weights->embeddings + token * dim;
    memcpy(x, embeddings, dim * sizeof(*x));

    for (unsigned long long layer = 0; layer < params->n_layers;layer++)
    {
        rms_norm(runstate->xb, x, weights->attn_norm + layer * dim, dim, eps);

        int layer_offset = layer * params->max_seq_len * kv_dim;
        runstate->k = runstate->key_cache + layer_offset + pos * kv_dim;
        runstate->v = runstate->value_cache + layer_offset + pos * kv_dim;

        matmul(runstate->q, runstate->xb, weights->wq + layer * dim * dim, dim, dim);
        matmul(runstate->k, runstate->xb, weights->wk + layer * dim * kv_dim, dim, kv_dim);
        matmul(runstate->v, runstate->xb, weights->wv + layer * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i+2)
        {
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcs = cosf(val);
            float fsi = sinf(val);
            int rotational = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotational; v++)
            {
                float* vect = v == 0 ? runstate->q : runstate->k;
                float v0 = vect[i];
                float v1 = vect[i + 1];
                vect[i] = v0 * fcs - v1 * fsi;
                vect[i + 1] = v0 * fsi + v1 * fcs;
            }
        }

        int h;
        #pragma omp parallel for private(h)
        for (h=0; h < num_heads; h++)
        {
            float *q = runstate->q + h * head_size;
            float *att = runstate->att + h * params->max_seq_len;
            for (int t=0; t <= pos; t++)
            {
                float *k = runstate->key_cache + layer_offset + t * kv_dim + (h/kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++)
                {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);

            float *xb = runstate->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t=0; t <= pos; t++)
            {
                float *v = runstate->value_cache + layer_offset + t * kv_dim + (h/kv_mul) * head_size;
                for (int i = 0; i < head_size; i++)
                {
                    xb[i] += att[t] * v[i];
                }
            }
        }
        matmul(runstate->xb2, runstate->xb, weights->wo + layer * dim * dim, dim, dim);

        for (int i = 0; i < dim; i++)
        {
            x[i] += runstate->xb2[i];
        }

        rms_norm(runstate->xb, x, weights->post_attn_norm + layer * dim, dim, eps);

        matmul(runstate->hb, runstate->xb, weights->w1 + layer * dim * hidden_dim, dim, hidden_dim);
        matmul(runstate->hb, runstate->xb, weights->w3 + layer * dim * hidden_dim, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; i++)
        {
            float val = runstate->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= runstate->hb2[i];
            runstate->hb[i] = val;
        }

        matmul(runstate->xb, runstate->hb, weights->w2 + layer * dim * hidden_dim, hidden_dim, dim);

        for (int i = 0; i < dim; i++)
        {
            x[i] += runstate->xb[i];
        }
    }
    rms_norm(x,x,weights->layer_norm,dim,eps);

    matmul(runstate->logits, x, weights->wcls, params->dim, params->vocab_size);
    return runstate->logits;
}

void allocate_runstate(Runstate* runstate, Config* config)
{
    int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
    runstate->x = calloc(config->dim, sizeof(float));
    runstate->xb = calloc(config->dim, sizeof(float));
    runstate->xb2 = calloc(config->dim, sizeof(float));
    runstate->hb = calloc(config->hidden_dim, sizeof(float));
    runstate->hb2 = calloc(config->hidden_dim, sizeof(float));
    runstate->q = calloc(config->dim, sizeof(float));
    runstate->key_cache = calloc(config->n_layers * config->max_seq_len * kv_dim, sizeof(float));
    runstate->value_cache = calloc(config->n_layers * config->max_seq_len * kv_dim, sizeof(float));
    runstate->att = calloc(config->n_heads * config->max_seq_len, sizeof(float));
    runstate->logits = calloc(config->vocab_size, sizeof(float));
}

void free_runstate(Runstate* runstate)
{
    free(runstate->x);
    free(runstate->xb);
    free(runstate->xb2);
    free(runstate->hb);
    free(runstate->hb2);
    free(runstate->q);
    free(runstate->key_cache);
    free(runstate->value_cache);
    free(runstate->att);
    free(runstate->logits);
}

void map_weights(Weights *weights, Config *config, float *ptr, int shared_weights)
{
    int head_size = config->dim / config->n_heads;
    unsigned long long n_layers = config->n_layers;
    weights->embeddings = ptr;
    ptr += config->vocab_size * config->dim;
    weights->attn_norm = ptr;
    ptr += n_layers * config->dim;
    weights->wq = ptr;
    ptr += n_layers * config->dim * (config->n_heads * head_size);
    weights->wk = ptr;
    ptr += n_layers * config->dim * (config->n_kv_heads * head_size);
    weights->wv = ptr;
    ptr += n_layers * config->dim * (config->n_kv_heads * head_size);
    weights->wo = ptr;
    ptr += n_layers * (config->n_heads * head_size) * config->dim;
    weights->post_attn_norm = ptr;
    ptr += n_layers * config->dim;
    weights->w1 = ptr;
    ptr += n_layers * config->dim * config->hidden_dim;
    weights->w2 = ptr;
    ptr += n_layers * config->hidden_dim * config->dim;
    weights->w3 = ptr;
    ptr += n_layers * config->dim * config->hidden_dim;
    weights->layer_norm = ptr;
    ptr += config->dim;
    ptr += config->max_seq_len * head_size / 2;
    ptr += config->max_seq_len * head_size / 2;
    weights->wcls = shared_weights ? weights->embeddings : ptr;
}

void read_model(char *filename, Config *config, Weights *weights, int *fd, float **data, ssize_t *size)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        _perror("Could not open Model file ");
        exit(1);
    }
    if (fread(config, sizeof(config), 1, file) != 1)
    {
        _perror("Could not read config from Model file\n");
        exit(1);
    }
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    fseek(file,0,SEEK_END);
    *size = ftell(file);
    fclose(file);

    *fd = open(filename, O_RDONLY);
    if (*fd == -1)
    {
        _perror("Could not open Model file\n");
        exit(1);
    }
    *data = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED)
    {
        _perror("Could not mmap Model File\n");
        exit(1);
    }
    float *weights_ptr = *data + sizeof(Config) / sizeof(float);
    map_weights(weights, config, weights_ptr, shared_weights);
}

void model(Transformer *transformer, char *filename)
{
    read_model(filename,&transformer->config, &transformer->weights, &transformer->fd, &transformer->data, &transformer->size);
    allocate_runstate(&transformer->runstate, &transformer->config);
}

void free_model(Transformer *transformer)
{
    if (transformer->data != MAP_FAILED)
    {
        munmap(transformer->data, transformer->size);
    }
    if (transformer->fd != -1)
    {
        close(transformer->fd);
    }
    free_runstate(&transformer->runstate);
}

int sample_argmax(float *prob, int argmax)
{
    int index = 0;
    float max = prob[0];
    for (int i = 1; i < argmax; i++) {
        if (prob[i] > max) {
            max = prob[i];
            index = i;
        }
    }
    return index;
}

int sample_multinomial(float *prob, int size, float coin)
{
    float cdf = 0.0f;
    for (int i = 0; i < size; i++) {
        cdf += prob[i];
        if (coin < cdf) {
            return i;
        }
    }
    return size-1;
}

int compare(const void *a, const void *b) {
    ProbIndex *a_ = (ProbIndex *) a;
    ProbIndex *b_ = (ProbIndex *) b;
    if (a_->prob < b_->prob) return 1;
    if (a_->prob > b_->prob) return -1;
    return 0;
}

int topp(float* prob, int size, float topp, ProbIndex* probindex, float coin)
{
    int n = 0;
    const float cutoff = (1.0f - topp) / (size-1);
    for (int i = 0; i < size; i++) {
        if(prob[i] >= cutoff)
        {
            probindex[n].index = i;
            probindex[n].prob = prob[i];
            n++;
        }
    }
    qsort(prob, n, sizeof(ProbIndex), compare);

    float cumulative_prob = 0.0f;
    int idx = n - 1;
    for (int i = 0; i < n; i++) {
    cumulative_prob += probindex[i].prob;
        if(cumulative_prob > topp)
        {
            idx = i;
            break;
        }
    }
    float random = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <=idx; i++) {
        cdf += probindex[i].prob;
        if (random < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[idx].index;
}

unsigned int random_u32(unsigned long long *state)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state)
{
    return (random_u32(state) >> 8 ) / 16777216.0f;
}

long time_in_ms()
{
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}


int sample(Sampler *s, float *logits)
{
    int next;
    if (s->temperature == 0.0f)
    {
        next = sample_argmax(logits, s->vocab_size);
    }else {
        for (int q = 0; q < s->vocab_size; q++)
        {
            logits[q] = s->temperature;
        }
        softmax(logits, s->vocab_size);
        float coin = random_f32(&s->rng_state);
        if (s->topp <= 0 || s->topp >= 1)
        {
            next = sample_multinomial(logits, s->vocab_size, coin);
        }else {
            next = topp(logits, s->vocab_size, s->topp, s->probindex, coin);
        }
    }
    return next;
}

void sampler(Sampler* s, int vocab_size, float temperature, float topp, unsigned long long rng_seed) 
{
    s->vocab_size = vocab_size;
    s->temperature = temperature;
    s->topp = topp;
    s->rng_state = rng_seed;
    s->probindex = malloc(s->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* s) 
{
    free(s->probindex);
}