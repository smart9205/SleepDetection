#include <stdio.h>
#include<sys/time.h>

#define TIME
#ifdef TIME
    #define TIME_START(name) \
        struct timeval name##start, name##end; \
        gettimeofday(&name##start, NULL);
    
    #define TIME_END(name) \
        gettimeofday(&name##end, NULL); \
        printf("\033[34m%s cost time: %.3f ms\033[0m\n", #name, (name##end.tv_sec - name##start.tv_sec) * 1000.0 + (name##end.tv_usec - name##start.tv_usec) / 1000.0);
#else
    #define TIME_START(name) {}; 
    #define TIME_END(name) {};
#endif // TIME
