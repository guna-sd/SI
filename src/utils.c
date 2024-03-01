#include "../include/shell.h"

static char currentDirectory[1024];
static char inputbuffer[INPUT_SIZE];

void initial_screen(void)
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
    snprintf(_path, sizeof(_path), "%s/.local/share/SI/.si_history", getenv("HOME"));
}
void _perror(char *ErrorMessage)
{
    fprintf(stderr, BRED);
    perror(ErrorMessage);
    fprintf(stderr, BWHITE);
}


char *_input(void) {
    char *input = malloc(INPUT_SIZE);
    if (input == NULL) {
        perror("Error: could not allocate memory for input");
        exit(EXIT_FAILURE);
    }
    if (fgets(input, INPUT_SIZE, stdin) == NULL)
    {
        _perror("Error: could not read input ^c from stdin");
        exit(EXIT_FAILURE);
    }

    return input;
}


void _store(char *args[]) {
    int total_length = 0;
    for (int i = 0; args[i] != NULL; i++) {
        total_length += strlen(args[i]) + 1;
    }
    char *input = malloc(total_length+1);
    if (input == NULL) {
        _perror("Error: could not allocate memory for input");
        return;
    }
    input[0] = '\0';
    for (int i = 0; args[i] != NULL; i++) {
        strcat(input, args[i]);
        strcat(input, " ");
    }

    int history_file = open(_path, O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (history_file == -1) {
        _perror("Error: could not open or create .si_history file");
        return;
    }
    if (flock(history_file, LOCK_EX) == -1) {
        _perror("Error: could not lock history file");
        close(history_file);
        return;
    }

    if (write(history_file, input, strlen(input)) == -1 || write(history_file, "\n", 1) == -1) {
        _perror("Error: could not write to history file");
        flock(history_file, LOCK_UN);
        close(history_file);
        return;
    }

    if (flock(history_file, LOCK_UN) == -1) {
        _perror("Error: could not unlock history file");
    }
    close(history_file);
}

void _clr_history(void) {
    FILE *history_file = fopen(_path, "w");
    if (history_file == NULL) {
        _perror("Error: could not open or create history file");
        return;
    }
    fclose(history_file);
    printf("History file cleared.\n");
}

void _history(void) {
    if (access(_path, F_OK) == -1) {
        fprintf(stderr, "Error: .si_history file does not exist\n");
        return;
    }

    FILE *history_file = fopen(_path, "r");
    if (history_file == NULL) {
        fprintf(stderr, "Error: Could not open .si_history file\n");
        return;
    }

    printf("Command history:\n");
    char buffer[1024];
    int line_count = 0;
    while (fgets(buffer, sizeof(buffer), history_file) != NULL) {
        printf("%d.%s", ++line_count, buffer);
    }

    fclose(history_file);
}

void prompt(void)
{
    char hostn[1024] = "";
    gethostname(hostn, sizeof(hostn));
    if ((getcwd(currentDirectory, sizeof(currentDirectory))))
    {
        printf(BGREEN"%s@%s:"BBLUE"[%s]"BWHITE"$!> ", getenv("LOGNAME"), hostn,currentDirectory);
        fflush(stdout);
    }
}

void _help(void) {
    printf("\n");
    printf(BBLUE"Welcome to Shell intelligence...!\n\n");
    printf("Available commands:\n");
    printf("\t1. help: Display this help message\n");
    printf("\t2. history: Display command history\n");
    printf("\t3. exit: Exit the shell\n");
    printf("\n This Shell is integrated with a llm (large language model) that is able to chat with developers and execute commands...\n\n"BWHITE);
}

void _out() {
    printf(BRED"\nExiting...\n\n"RESET);
    exit(EXIT_SUCCESS);
}



long time_in_ms()
{
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void bprintf(char *rbytes)
{
    if (rbytes == NULL)
    {
        return;
    }
    if (rbytes[0] == '\0')
    {
        return;
    }
    if (rbytes[1] == '\0')
    {
        unsigned char buf = rbytes[0];
        if (!(isprint(buf) || isspace(buf)))
        {
            return;
        }
    }
    printf(BWHITE"%s\n"BWHITE, rbytes);
}

void matmul(float* out, float* x, float* w, int n, int d) {
    int i, j;
    #pragma omp parallel for private(i, j)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        out[i] = val;
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}


void rmsnorm(float *o, float *x, float *w, int size)
{
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += x[i] * x[i];
    }
    sum /= size;
    sum += 1e-5f;
    sum = 1.0f / sqrtf(sum);
    for (int i = 0; i < size; i++) {
        o[i] = w[i] * (sum * x[i]);
    }
}

int compare_tokens(const void *a, const void *b) {
    return strcmp(((Tokens*)a)->token, ((Tokens*)b)->token);
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

int sample(Sampler *sampler, float *logits)
{
    int next;
    if (sampler->temperature == 0.0f)
    {
        next = sample_argmax(logits, sampler->vocab_size);
    }else {
        for (int q = 0; q < sampler->vocab_size; q++)
        {
            logits[q] = sampler->temperature;
        }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1)
        {
            next = sample_multinomial(logits, sampler->vocab_size, coin);
        }else {
            next = topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

