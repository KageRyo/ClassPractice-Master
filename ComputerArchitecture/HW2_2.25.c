#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

/*
 * Intel Core i5-14500
 *
 * L1 Data Cache = 32KB~48KB/core -> ARRAY_MIN = 4096 (= 16KB, 遠小於 L1)
 * L2 Cache      = 11.5MB total   -> 需測到 16MB+ 才容易看出 L2/L3 過渡
 * L3 Cache      = 24MB shared    -> 需測到 48MB+ 才明顯看出 L3 miss
 * RAM           = 8GB            -> 不需測到超過 RAM
 * TLB           = ~256KB-6MB 覆蓋範圍 (4KB pages)
 *
 * ARRAY_MAX = 16M integers = 64MB, 足以超過 L3 24MB
 * 每組測 4 秒，總執行時間約 20-30 分鐘
 */

#define ARRAY_MIN  (4096)              // 16KB, 遠小於 L1
#define ARRAY_MAX  (16 * 1024 * 1024)  // 16M ints = 64MB, 超過 L3 24MB
#define MEASURE_SECS  4.0              // 每組收集秒數

int x[ARRAY_MAX];
volatile int benchmark_sink = 0;

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

    fprintf(stderr, "=== Memory Benchmark for i5-14500 ===\n");
    fprintf(stderr, "L1=32~48KB  L2=11.5MB  L3=24MB  RAM=8GB\n");
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
                    benchmark_sink ^= nextstep;
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
                    benchmark_sink ^= index;
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
    if (benchmark_sink == -1)
        fprintf(stderr, "benchmark_sink=%d\n", benchmark_sink);
    return 0;
}