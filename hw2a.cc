#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <emmintrin.h>
#include <iostream>
#include <png.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double interval_x, interval_y;

int iters, width, height;
int *image;
double lower, upper, left, right;
const double zero = 0;
const double twoConst = 2.0;
int current_row = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

union simd {
  alignas(16) double vs[2];
  __m128d v;
};

void write_png(const char *filename, int iters, int width, int height,
               const int *buffer) {
    FILE *fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                color[0] = 240;
                color[1] = color[2] = p % 16 * 16;
                } else {
                color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void *worker2(void *args) {
    int curX = 0;
    int local_row, local;
    int repeats[2];
    double tmp, ys;
    int is[2];

    simd length, a, b, x;

    __m128d y, sqrtA, sqrtB, tempA;
    __m128d two = _mm_load1_pd(&twoConst);

    while (true) {
        pthread_mutex_lock(&mutex);
        if (current_row >= height) {
            pthread_mutex_unlock(&mutex);
            break;
        }

        local_row = current_row++;
        pthread_mutex_unlock(&mutex);

        
        local = local_row * width;
        ys = local_row * interval_y + lower;
        y = _mm_load_pd1(&ys);

        is[0] = curX++;
        is[1] = curX++;
        repeats[0] = repeats[1] = 0;
        x.v = _mm_setr_pd(is[0] * interval_x + left, is[1] * interval_x + left);
        a.v = b.v = _mm_setzero_pd();


        while (is[0] < width && is[1] < width) {
            while (true) {
                sqrtA = _mm_mul_pd(a.v, a.v);
                sqrtB = _mm_mul_pd(b.v, b.v);
                tempA = a.v;
                length.v = _mm_add_pd(sqrtA, sqrtB);

                if (length.vs[0] >= 4.0 || length.vs[1] >= 4.0) break;
                ++repeats[0];
                ++repeats[1];

                a.v = _mm_add_pd(_mm_sub_pd(sqrtA, sqrtB), x.v);
                b.v = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(tempA, b.v), two), y);

                if (repeats[0] >= iters || repeats[1] >= iters) break;
            }

            if (length.vs[0] >= 4.0 || repeats[0] >= iters) {
                image[local + is[0]] = repeats[0];
                is[0] = curX++;
                repeats[0] = 0;
                a.vs[0] = b.vs[0] = 0;
                x.vs[0] = is[0] * interval_x + left;
            }

            if (length.vs[1] >= 4.0 || repeats[1] >= iters) {
                image[local + is[1]] = repeats[1];
                is[1] = curX++;
                repeats[1] = 0;
                a.vs[1] = b.vs[1] = 0;
                x.vs[1] = is[1] * interval_x + left;
            }
            x.v = _mm_load_pd(x.vs);
            a.v = _mm_load_pd(a.vs);
            b.v = _mm_load_pd(b.vs);
        }

        if (is[0] < width) {
            while(repeats[0] < iters && a.vs[0] * a.vs[0] + b.vs[0] * b.vs[0] < 4){
                tmp = a.vs[0];
                a.vs[0] = a.vs[0] * a.vs[0] - b.vs[0] * b.vs[0] + x.vs[0];
                b.vs[0] = 2 * tmp * b.vs[0] + ys;
                ++repeats[0];
            }
            image[local + is[0]] = repeats[0];
        } else {
            while(repeats[1] < iters && a.vs[1] * a.vs[1] + b.vs[1] * b.vs[1] < 4){
                tmp = a.vs[1];
                a.vs[1] = a.vs[1] * a.vs[1] - b.vs[1] * b.vs[1] + x.vs[1];
                b.vs[1] = 2 * tmp * b.vs[1] + ys;
                ++repeats[1];
            }
            image[local + is[1]] = repeats[1];
        }
        curX = 0;
    }
    return NULL;
}

int main(int argc, char **argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

    int num = CPU_COUNT(&cpu_set);
    pthread_t threads[num];
    const char *filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    interval_y = ((upper - lower) / height);
    interval_x = ((right - left) / width);

    /* allocate memory for image */
    image = (int *)malloc(width * height * sizeof(int));

    /* mandelbrot set */
    for (int k = 0; k < num; k++) {
        pthread_create(&threads[k], NULL, worker2, NULL);
    }

    for (int k = 0; k < num; k++) {
        pthread_join(threads[k], NULL);
    }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}

