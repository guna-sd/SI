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
