#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

#define PAGE_BYTES 4096
#define PAGE_ELEMS (PAGE_BYTES / (int)sizeof(int))
#define TLB_ASSOC_MEASURE_SECS 3.0
#define TLB_ASSOC_MAX_WAYS 24

int x[ARRAY_MAX];
volatile int benchmark_sink = 0;

void pin_thread_to_core0(void) {
    HANDLE thread = GetCurrentThread();
    DWORD_PTR prev_mask = SetThreadAffinityMask(thread, (DWORD_PTR)1);
    if (prev_mask == 0) {
        fprintf(stderr, "Warning: failed to set CPU affinity.\n");
    }
}

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

double measure_pointer_chase_ns_per_access(int start_index, int access_count, double measure_secs) {
    int i, nextstep;
    double sec0, sec1, sec;
    unsigned long long steps;
    unsigned long long k;

    steps = 0;
    sec0 = get_seconds();
    do {
        nextstep = start_index;
        do {
            nextstep = x[nextstep];
        } while (nextstep != start_index);
        benchmark_sink ^= nextstep;
        steps += 1;
        sec1 = get_seconds();
    } while ((sec1 - sec0) < measure_secs);
    sec = sec1 - sec0;

    /* Time-loop overhead subtraction. */
    sec0 = get_seconds();
    for (k = 0; k < steps; k++) {
        nextstep = start_index;
        for (i = 0; i < access_count; i++) {
            nextstep += 1;
        }
        benchmark_sink ^= nextstep;
    }
    sec1 = get_seconds();
    sec -= (sec1 - sec0);

    if (sec < 0.0)
        sec = 0.0;
    return (sec * 1e9) / ((double)steps * (double)access_count);
}

void run_tlb_associativity_benchmark(const char *out_csv) {
    FILE *fp;
    int stride_pages_list[] = {16, 32, 64, 128};
    int stride_count = (int)(sizeof(stride_pages_list) / sizeof(stride_pages_list[0]));
    int ways, s, w;

    fp = fopen(out_csv, "w");
    if (!fp) {
        fprintf(stderr, "Warning: cannot open %s for writing, skip TLB associativity benchmark.\n", out_csv);
        return;
    }

    fprintf(stderr, "\n=== TLB Associativity Conflict Benchmark ===\n");
    fprintf(stderr, "Stride in page multiples: 16, 32, 64, 128 pages\n");
    fprintf(stderr, "Output: %s\n", out_csv);

    fprintf(fp, "ways");
    for (s = 0; s < stride_count; s++)
        fprintf(fp, ",%dp", stride_pages_list[s]);
    fprintf(fp, "\n");

    for (ways = 1; ways <= TLB_ASSOC_MAX_WAYS; ways++) {
        fprintf(stderr, "  TLB conflict ways = %d ...\n", ways);
        fprintf(fp, "%d", ways);

        for (s = 0; s < stride_count; s++) {
            int stride_pages = stride_pages_list[s];
            int stride_elems = stride_pages * PAGE_ELEMS;
            int start_index;
            double ns_per_access;

            for (w = 0; w < ways; w++) {
                int cur = w * stride_elems;
                int next = ((w + 1) % ways) * stride_elems;
                if (cur >= ARRAY_MAX || next >= ARRAY_MAX) {
                    fprintf(stderr, "Warning: index out of range in TLB benchmark, reduce settings.\n");
                    fclose(fp);
                    return;
                }
                x[cur] = next;
            }

            start_index = ((ways / 2) % ways) * stride_elems;
            ns_per_access = measure_pointer_chase_ns_per_access(start_index, ways, TLB_ASSOC_MEASURE_SECS);
            if (ns_per_access < 0.1)
                ns_per_access = 0.1;
            fprintf(fp, ",%.3f", ns_per_access);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

int main(int argc, char **argv) {
    int nextstep, i, index, stride, csize;
    unsigned long long steps;
    unsigned long long k;
    double loadtime, sec0, sec1, sec, lastsec;

    pin_thread_to_core0();

    if (argc > 1 && strcmp(argv[1], "--tlb-assoc-only") == 0) {
        run_tlb_associativity_benchmark("tlb_assoc_benchmark.csv");
        fprintf(stderr, "Done! TLB associativity CSV saved.\n");
        return 0;
    }

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
            int prev;

            prev = 0;
            for (index = stride; index < csize; index += stride) {
                x[prev] = index;
                prev = index;
            }
            x[prev] = 0;

            lastsec = get_seconds();
            do { sec0 = get_seconds(); } while (sec0 == lastsec);

            steps = 0;
            nextstep = 0;
            sec0 = get_seconds();
            do {
                for (i = stride; i != 0; i--) {
                    nextstep = 0;
                    do { nextstep = x[nextstep]; } while (nextstep != 0);
                    benchmark_sink ^= nextstep;
                }
                steps += 1;
                sec1 = get_seconds();
            } while ((sec1 - sec0) < MEASURE_SECS);
            sec = sec1 - sec0;

            sec0 = get_seconds();
            for (k = 0; k < steps; k++) {
                for (i = stride; i != 0; i--) {
                    index = 0;
                    do { index += stride; } while (index < csize);
                    benchmark_sink ^= index;
                }
            }
            sec1 = get_seconds();
            sec -= (sec1 - sec0);

            loadtime = (sec * 1e9) / ((double)steps * (double)csize);
            printf("%4.1f,", (loadtime < 0.1) ? 0.1 : loadtime);
        }
        printf("\n");
        fflush(stdout);
    }

    fprintf(stderr, "\nDone! Output saved to CSV.\n");
    run_tlb_associativity_benchmark("tlb_assoc_benchmark.csv");

    if (benchmark_sink == -1)
        fprintf(stderr, "benchmark_sink=%d\n", benchmark_sink);
    return 0;
}