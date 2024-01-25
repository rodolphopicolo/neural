#include <stddef.h>
#include <stdio.h>
#include <time.h>



long checkpoint(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    time_t s  = t.tv_sec;
    long ms = t.tv_nsec / 1.0e6;
    ms = ms + s * 1000;
    return ms;
}   