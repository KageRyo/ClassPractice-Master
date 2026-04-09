#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

/*
 * i7-10700
 *
 * L1 Data Cache = 32KB/core  -> ARRAY_MIN = 2048  (= 8KB, 遠小於 L1)
 * L2 Cache      = 256KB/core -> 需測到 512KB+ 才看出 L2 miss
 * L3 Cache      = 16MB shared -> 需測到 32MB+ 才看出 L3 miss
 * RAM           = 16GB       -> 不需測到超過 RAM
 * TLB           = ~256KB-6MB 覆蓋範圍 (4KB pages)
 *
 * ARRAY_MAX = 8M integers = 32MB, 足以超過 L3
 * 每組測 5 秒（而非 20 秒），總執行時間約 20-30 分鐘
 */

#define ARRAY_MIN  (2048)            // 8KB, 遠小於 L1 32KB
#define ARRAY_MAX  (8 * 1024 * 1024) // 8M ints = 32MB, 超過 L3 16MB
#define MEASURE_SECS  5.0            // 每組收集秒數

int x[ARRAY_MAX];

double get_seconds() {
    LARGE_INTEGER cnt, freq;
    QueryPerformanceCounter(&cnt);
    QueryPerformanceFrequency(&freq);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}

void label(int bytes) {
    if (bytes < 1024)
        printf("%dB,", bytes);
    else if (bytes < 1048576)
        printf("%dK,", bytes / 1024);
    else if (bytes < 1073741824)
        printf("%dM,", bytes / 1048576);
    else
        printf("%dG,", bytes / 1073741824);
}

int main() {
    int nextstep, i, index, stride, csize;
    double steps, tsteps;
    double loadtime, sec0, sec1, sec, lastsec;

    fprintf(stderr, "=== Memory Benchmark for i7-10700 ===\n");
    fprintf(stderr, "L1=32KB  L2=256KB  L3=16MB  RAM=16GB\n");
    fprintf(stderr, "Each config measured for %.0f seconds.\n", MEASURE_SECS);
    fprintf(stderr, "Please close other applications for best results.\n");
    fprintf(stderr, "Estimated time: 20-30 minutes\n\n");

    /* CSV header: stride values */
    printf(" ,");
    for (stride = 1; stride <= ARRAY_MAX / 2; stride *= 2)
        label(stride * (int)sizeof(int));
    printf("\n");

    for (csize = ARRAY_MIN; csize <= ARRAY_MAX; csize *= 2) {
        if (csize * (int)sizeof(int) < 1048576)
            fprintf(stderr, "  Testing array size = %dKB ...\n",
                    csize * (int)sizeof(int) / 1024);
        else
            fprintf(stderr, "  Testing array size = %dMB ...\n",
                    csize * (int)sizeof(int) / 1048576);

        label(csize * (int)sizeof(int));

        for (stride = 1; stride <= csize / 2; stride *= 2) {

            for (index = 0; index < csize; index += stride)
                x[index] = index + stride;
            x[index - stride] = 0;

            lastsec = get_seconds();
            do { sec0 = get_seconds(); } while (sec0 == lastsec);

            steps = 0.0;
            nextstep = 0;
            sec0 = get_seconds();
            do {
                for (i = stride; i != 0; i--) {
                    nextstep = 0;
                    do { nextstep = x[nextstep]; } while (nextstep != 0);
                }
                steps += 1.0;
                sec1 = get_seconds();
            } while ((sec1 - sec0) < MEASURE_SECS);
            sec = sec1 - sec0;

            tsteps = 0.0;
            sec0 = get_seconds();
            do {
                for (i = stride; i != 0; i--) {
                    index = 0;
                    do { index += stride; } while (index < csize);
                }
                tsteps += 1.0;
                sec1 = get_seconds();
            } while (tsteps < steps);
            sec -= (sec1 - sec0);

            loadtime = (sec * 1e9) / (steps * (double)csize);
            printf("%4.1f,", (loadtime < 0.1) ? 0.1 : loadtime);
        }
        printf("\n");
        fflush(stdout);
    }

    fprintf(stderr, "\nDone! Output saved to CSV.\n");
    return 0;
}