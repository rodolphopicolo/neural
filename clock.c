#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



long checkpoint(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    time_t s  = t.tv_sec;
    long ms = t.tv_nsec / 1.0e6;
    ms = ms + s * 1000;
    return ms;
}


char *current_time_to_string(){
    time_t rawtime;
    struct tm *time_info;
    char *buffer = malloc(19);
    time( &rawtime );
    time_info = localtime( &rawtime );
    strftime(buffer, 19, "%Y-%m-%d-%H-%M-%S", time_info);
    return buffer;
}

void current_timestamp(){
    time_t t;
    time(&t);
    printf("\nThis program has been writeen at (date and time): %s", ctime(&t));
}