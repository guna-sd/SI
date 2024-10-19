#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <curl/curl.h>
#include <jansson.h>


void test_get_input() {
    const char *test_input = "Hello World\n";
    FILE *stdin_backup = stdin;

    FILE *test_input_stream = fmemopen((void *)test_input, strlen(test_input), "r");
    stdin = test_input_stream;

    char *result = get_input();
    assert(result != NULL);
    assert(strcmp(result, "Hello World") == 0);

    free(result);
    fclose(test_input_stream);
    stdin = stdin_backup;
}

void test_handle_builtin_commands() {
    assert(handle_builtin_commands("exit") == 1);
    
    assert(handle_builtin_commands("help") == 0);
    
    assert(handle_builtin_commands("cd") == 0);

    assert(handle_builtin_commands("cd non_existent_dir") == 0); // Should return 0 but print an error
}

int main() {
    test_get_input();
    test_handle_builtin_commands();
    printf("All tests passed!\n");
    return 0;
}
