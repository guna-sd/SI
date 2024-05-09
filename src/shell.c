
#include "../include/shell.h"

pid_t pid;
static pid_t SHELL_PGID, SHELL_PID;
static int shell_is_interactive, _terminal;
struct termios _shell;
struct sigaction act_child, act_interrupt;
static Transformer transformer;
static Tokenizer tokenizer;
static Sampler sampler;

static void init()
{
    _terminal = STDIN_FILENO;
    shell_is_interactive = isatty(_terminal);
    SHELL_PID = getpid();
    if (shell_is_interactive)
    {
        while (tcgetpgrp(_terminal) != (SHELL_PGID=getpgrp()))
            kill(SHELL_PID,SIGTTIN);

        act_child.sa_handler = signalHandler_child;
        act_interrupt.sa_handler = signalHandler_interrupt;

        sigaction(SIGCHLD, &act_child, 0);
        sigaction(SIGINT, &act_interrupt, 0);


        setpgid(SHELL_PID, SHELL_PID);
        SHELL_PGID = getpgrp();
        if (SHELL_PID != SHELL_PGID)
        {
            _perror("Error, the shell is not process group leader");
            exit(1);
        }
        tcsetpgrp(_terminal, SHELL_PGID);
        tcgetattr(_terminal, &_shell);
    }
    else 
    {
        _perror("Error : Could not set SHELL as interactive...");
        exit(1);
    }
    build_transformer(&transformer, _model);
    build_tokenizer(&tokenizer, _tok);
}

static void signalHandler_interrupt(int signal)
{
    if (kill(pid, SIGTERM) == 0)
    {
        printf(BGREEN "Interrupted pid %d\n", pid);
    }
    else
    {
		printf("\n");
    }
}

static void signalHandler_child(int signal)
{
    while (waitpid(-1, NULL, WNOHANG) > 0) 
    {

    }
}

int _cd(char *args[]) {
    if (args[1] == NULL) {
        if (chdir(getenv("HOME")) == -1) {
            _perror("Error, could not change directory");
            return CMD_FAILURE;
        }
        return CMD_SUCCESS;
    } else {
        if (chdir(args[1]) == -1) {
            _perror("Error, No directory such directory...");
            return CMD_FAILURE;
        }
        return CMD_SUCCESS;
    }
    return CMD_SUCCESS;
}


void _exec(char **args)
{
    pid = fork();
    if (pid == -1)
    {
        _perror("Error, could not fork child process not created...");
        return;
    }
    if (pid == 0)
    {
        if (execvp(args[0], args) == -1)
        {
            _perror("Error, could not execute command...");
            kill(getpid(), SIGTERM);
        }
    }
    else
    {
        waitpid(pid, NULL, 0);
    }
    return;
}

void _exec_bg(char **args) {
    pid = fork();

    if (pid == -1) {
        _perror("Error, could not fork child process not created...");
        return;
    } else if (pid == 0) {
        if (execvp(args[0], args) == -1) {
            _perror("Error, could not execute command...");
            exit(EXIT_FAILURE);
        }
    } else {
        printf("Started background process with PID %d\n", pid);
    }
}

void _io(char *args[], char *outputFile, int option) {
    int fd;
    pid_t pid = fork();
    if (pid == -1) {
        _perror("Error: fork failed");
        return;
    }
    if (pid == 0) {
        if (option == 0) {
            fd = open(outputFile, O_CREAT | O_TRUNC | O_WRONLY, 0600);
        } else if (option == 1) {
            fd = open(outputFile, O_CREAT | O_APPEND | O_WRONLY, 0600);
        } else {
            fprintf(stderr, "Error: Invalid redirection type\n");
            exit(EXIT_FAILURE);
        }
        if (fd == -1) {
            perror("Error: failed to open output file");
            exit(EXIT_FAILURE);
        }
        if (dup2(fd, STDOUT_FILENO) == -1) {
            perror("Error: failed to redirect stdout");
            exit(EXIT_FAILURE);
        }
        close(fd);
        if (execvp(args[0], args) == -1) {
            perror("Error: failed to execute command");
            exit(EXIT_FAILURE);
        }
    } else {
        waitpid(pid, NULL, 0);
    }
}
void _pipe(char *args[]) {
    int num_cmds = 0;
    char *command[256];
    int err = -1;

    for (int i = 0; args[i] != NULL; ++i) {
        if (strcmp(args[i], "|") == 0) {
            num_cmds++;
        }
    }
    num_cmds++;

    int pipe_fds[num_cmds - 1][2];

    for (int i = 0, j = 0; args[j] != NULL; ++i) {
        int k = 0;

        while (args[j] != NULL && strcmp(args[j], "|") != 0) {
            command[k++] = args[j++];
        }
        command[k] = NULL;

        if (i < num_cmds - 1) {
            if (pipe(pipe_fds[i]) == -1) {
                _perror("pipe");
                return;
            }
        }

        pid = fork();
        if (pid == -1) {
            _perror("fork");
            return;
        }

        if (pid == 0) {
            if (i > 0) {
                dup2(pipe_fds[i - 1][0], STDIN_FILENO);
                close(pipe_fds[i - 1][0]);
                close(pipe_fds[i - 1][1]);
            }

            if (i < num_cmds - 1) {
                dup2(pipe_fds[i][1], STDOUT_FILENO);
                close(pipe_fds[i][0]);
                close(pipe_fds[i][1]);
            }

            if (execvp(command[0], command) == err) {
                _perror("execvp");
                exit(EXIT_FAILURE);
            }
        } else {
            if (i > 0) {
                close(pipe_fds[i - 1][0]);
                close(pipe_fds[i - 1][1]);
            }

            while (args[j] != NULL && strcmp(args[j], "|") != 0) {
                j++;
            }

            if (args[j] != NULL) {
                j++;
            }
        }
    }

    for (int i = 0; i < num_cmds - 1; ++i) {
        close(pipe_fds[i][0]);
        close(pipe_fds[i][1]);
    }

    for (int i = 0; i < num_cmds; ++i) {
        waitpid(pid, NULL, 0);
    }
}

void _logical(char *args[]) {
    int i = 0;
    int status;

    while (args[i] != NULL) {
        int k = 0;
        char *command[256];

        while (args[i] != NULL && strcmp(args[i], "&&") != 0 && strcmp(args[i], "||") != 0) {
            command[k++] = args[i++];
        }
        command[k] = NULL;

        pid = fork();
        if (pid == -1) {
            _perror("Error: could not fork child process");
            return;
        }
        if (pid == 0) {
            if (execvp(command[0], command) == -1) {
                _perror("Error: could not execute command");
                exit(EXIT_FAILURE);
            }
        } else {
            waitpid(pid, &status, 0);

            if (args[i] != NULL) {
                if (strcmp(args[i], "&&") == 0) {
                    if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
                        break;
                    }
                } else if (strcmp(args[i], "||") == 0) {
                    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
                        break;
                    }
                }
                i++;  
            } else {
                break;
            }
        }
    }
}


int _cmdh(char *args[]) {
    int i = 0, j = 0, background = 0;
    char *args_aux[256];

    while (args[i] != NULL) {
        if (strcmp(args[i], "&") == 0 || strcmp(args[i], ">") == 0 || strcmp(args[i], ">>") == 0 || strcmp(args[i], "&&") == 0) {
            break;
        }
        args_aux[i] = args[i];
        i++;
    }

    if (strcmp(args[0], "exit") == 0) {
        _out();
    } else if (strcmp(args[0], "cd") == 0) {
        _cd(args);
    } else if (strcmp(args[0], "history") == 0) {
        _history();
    } else if (strcmp(args[0], "clear") == 0) {
        if(system("clear")) printf("\n");
    } else if (strcmp(args[0], "help") == 0) {
        _help();
    } else {
        while (args[j] != NULL && background == 0) {
            if (strcmp(args[j], "&") == 0) {
                background = 1;
                break;
            } else if (strcmp(args[j], "|") == 0) {
                _pipe(args);
                return CMD_SUCCESS;
            } else if (strcmp(args[j], "||") == 0) {
                _logical(args);
                return CMD_SUCCESS;
            } else if (strcmp(args[j], "&&") == 0) {
                _logical(args);
                return CMD_SUCCESS;
            } else if (strcmp(args[j], ">") == 0) {
                if (args[j + 1] == NULL) {
                    printf("Not enough input arguments\n");
                    return -1;
                }
                _io(args_aux, args[j + 1], 0);
                return 1;
            } else if (strcmp(args[j], ">>") == 0) {
                if (args[j + 1] == NULL) {
                    printf("Not enough input arguments\n");
                    return -1;
                }
                _io(args_aux, args[j + 1], 1);
                return 1;
            }
            j++;
        }
        args_aux[j] = NULL;
        if (background == 1) {
            _exec_bg(args_aux);
        } else {
            _exec(args_aux);
        }
    }
    return CMD_SUCCESS;
}

char generate(char *input)
{
    
}

int main() {
    char *args[256];
    char *input;
    char *token;
    init();
    build_sampler(&sampler, transformer.config.vocab_size, 1.0, 0.8, transformer.config.max_seq_len);
    initial_screen();
    
    while (1) {
        prompt();
        input = _input();
        
        token = strtok(input, " \t\n");
        int i = 0;
        while (token != NULL) {
            args[i++] = token;
            token = strtok(NULL, " \t\n");
        }
        args[i] = NULL;
        if (args[0] != NULL) {
            _cmdh(args);
        }
    }
    return 0;
}

