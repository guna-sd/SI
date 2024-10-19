#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <curl/curl.h>
#include <jansson.h>

#define MAX_INPUT_SIZE 1024

struct curl_buffer {
    char *data;
    size_t size;
};

size_t write_callback(void *ptr, size_t size, size_t nmemb, struct curl_buffer *buffer) {
    size_t total_size = size * nmemb;
    char *new_data = realloc(buffer->data, buffer->size + total_size + 1);
    if (new_data == NULL) {
        fprintf(stderr, "Not enough memory to allocate buffer\n");
        return 0;
    }

    buffer->data = new_data;
    memcpy(&(buffer->data[buffer->size]), ptr, total_size);
    buffer->size += total_size;
    buffer->data[buffer->size] = '\0';
    return total_size;
}

char *get_input() {
    char *input = malloc(MAX_INPUT_SIZE);
    if (!input) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    printf("\e[1;92m%s@%s:\e[1;94m[%s]\e[1;97mS!> ", getlogin(), getenv("HOSTNAME"), getcwd(NULL, 0));
    fgets(input, MAX_INPUT_SIZE, stdin);
    input[strcspn(input, "\n")] = '\0';
    return input;
}

void ask_llm(const char *prompt) {
    CURL *curl;
    CURLcode res;

    char json_payload[512];
    snprintf(json_payload, sizeof(json_payload),
             "{\"model\": \"SI\", \"prompt\": \"%s\"}", prompt);

    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if (curl) {
        struct curl_slist *headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/generate");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);

        struct curl_buffer response_buffer;
        response_buffer.data = malloc(1);
        response_buffer.size = 0;

        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_buffer);

        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        } else {
            //printf("\e[1;93mRaw LLM Response: \e[1;97m%s\n", response_buffer.data);

            char *line = strtok(response_buffer.data, "\n");
            while (line) {
                json_error_t error;
                json_t *root = json_loads(line, 0, &error);
                if (!root) {
                    fprintf(stderr, "Error parsing JSON: %s\n", error.text);
                } else {
                    json_t *response = json_object_get(root, "response");
                    
                    if (json_is_string(response)) {
                        printf("%s", json_string_value(response));
                    }

                    json_decref(root);
                }
                line = strtok(NULL, "\n");
            }
            printf("\n");
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        free(response_buffer.data);
    }
    curl_global_cleanup();
}



int handle_builtin_commands(char *input) {
    if (strcmp(input, "exit") == 0) {
        printf("Exiting shell\n");
        return 1;
    }

    if (strcmp(input, "help") == 0) {
        printf("Builtin commands:\n");
        printf("  cd [directory] - Change the current directory\n");
        printf("  exit - Exit the shell\n");
        printf("  help - Show this help message\n");
        return 0;
        }

    if (strncmp(input, "cd", 2) == 0) {
        char *dir = strtok(input + 3, " ");
        if (dir == NULL || strlen(dir) == 0) {
            chdir(getenv("HOME"));
        } else {
            if (chdir(dir) != 0) {
                perror("cd");
            }
        }
        return 0;
    }

    return -1;
}

int main() {
    while (1) {
        char *input = get_input();

        int builtin_status = handle_builtin_commands(input);
        if (builtin_status == 1) {
            free(input);
            break;
        } else if (builtin_status == 0) {
            free(input);
            continue;
        }

        ask_llm(input);

        free(input);
    }

    return 0;
}