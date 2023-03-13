#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <emmintrin.h>
#include <math.h>

int *displs;
int rank, size;
void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        int temp = height - 1 - y;
        // int box = (pngpls[(temp % size)] + int(temp / size))*width;
        // printf("%d\n", temp);
        for (int x = 0; x < width; ++x) {
            int p = buffer[displs[(temp % size)] + int(temp / size) * width + x];
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
union simd {
    alignas(16) double vs[2];
    __m128d v;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    int num = CPU_COUNT(&cpu_set);

    /* argument parsing */
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);
    int row_per_proc = height / size;
    double interval_x = ((right - left) / width);
    double interval_y = ((upper - lower) / height);
    
    __m128d two = _mm_set1_pd(2.0);

    int remainder = height % size;
    displs = (int*)malloc(size * sizeof(int));
    int *rcount = (int*)malloc(size * sizeof(int));
    int *image = NULL;
    int num_item = row_per_proc * width;
    
    #pragma omg parallel for schedule(dynamic)
    for(int i = remainder; i < size; ++i){
        displs[i] = (row_per_proc * i + remainder) * width;
        rcount[i] = num_item;
    }

    num_item += width;
    #pragma omg parallel for schedule(dynamic)
    for(int i = 0; i < remainder; ++i){
        displs[i] = i * (row_per_proc + 1) * width;
        rcount[i] = num_item;
    }
    
    if(rank < remainder) {
        ++row_per_proc;
    }

    int* local_image = (int*)malloc(row_per_proc * width * sizeof(int));

    if(rank == 0) image = (int*)malloc(width * height * sizeof(int));
    
    
    #pragma omp parallel num_threads(num)
    {
        // int local;
        double tmp;
        int local_col;
        int repeats[2];
        double as[1], bs[1], xs[1], ys[1];
        int flag = 0;
        int local_row;
        simd length, x, y, a, b;
        __m128d tempA;      

        
        #pragma omp for schedule(dynamic, 5) collapse(2)
        for(int i = 0; i < row_per_proc; ++i){
            for(local_col = 0; local_col < width - 1; local_col += 2){
                local_row = rank + i * size;
                ys[0] = local_row * interval_y + lower;
                y.v = _mm_load_pd1(&ys[0]);
                x.v = _mm_setr_pd(local_col * interval_x + left, (local_col + 1) * interval_x + left);
                a.v = _mm_setzero_pd();
                b.v = _mm_setzero_pd();
                repeats[0] = 0;
                repeats[1] = 0;
                while(true){
                    tempA = a.v;
                    a.v = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(a.v, a.v), _mm_mul_pd(b.v, b.v)), x.v);
                    b.v = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(tempA, b.v), two), y.v);
                    length.v = _mm_add_pd(_mm_mul_pd(a.v, a.v), _mm_mul_pd(b.v, b.v));
                    
                    ++repeats[0];
                    ++repeats[1];
                    
                    if(repeats[0] >= iters || length.vs[0] >= 4.0){
                        local_image[i * width + local_col] = repeats[0];
                        flag = 1;
                        break;
                    }

                    if(repeats[1] >= iters || length.vs[1] >= 4.0){
                        local_image[i * width + local_col + 1] = repeats[1];
                        flag = 0;
                        break;
                    }
                }

                while(repeats[flag] < iters && length.vs[flag] < 4){
                    tmp = a.vs[flag];
                    a.vs[flag] = a.vs[flag] * a.vs[flag] - b.vs[flag] * b.vs[flag] + x.vs[flag];
                    b.vs[flag] = 2 * tmp * b.vs[flag] + y.vs[flag];
                    length.vs[flag] = a.vs[flag] * a.vs[flag] + b.vs[flag] * b.vs[flag];
                    ++repeats[flag];
                }
                local_image[i * width + local_col + flag] = repeats[flag];
            }
        }

        if(width & 1){
            local_col = width - 1;
            xs[0] = local_col * interval_x + left;
            x.v = _mm_load_pd1(&xs[0]);

            #pragma omp for schedule(dynamic, 2)
            for(int i = 0; i < row_per_proc - 1; i += 2){
                local_row = rank + i * size;
                repeats[0] = 0;
                repeats[1] = 0;
                y.v = _mm_setr_pd(local_row * interval_y + lower, (local_row + size) * interval_y + lower);
                a.v = _mm_setzero_pd();
                b.v = _mm_setzero_pd();

                while(true){
                    tempA = a.v;
                    a.v = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(a.v, a.v), _mm_mul_pd(b.v, b.v)), x.v);
                    b.v = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(tempA, b.v), two), y.v);
                    length.v = _mm_add_pd(_mm_mul_pd(a.v, a.v), _mm_mul_pd(b.v, b.v));
                    
                    ++repeats[0];
                    ++repeats[1];
                    
                    if(repeats[0] >= iters || length.vs[0] >= 4.0){
                        local_image[i * width + local_col] = repeats[0];
                        flag = 1;
                        break;
                    }

                    if(repeats[1] >= iters || length.vs[1] >= 4.0){
                        local_image[(i+1) * width + local_col] = repeats[1];
                        flag = 0;
                        break;
                    }
                }

                while(repeats[flag] < iters && length.vs[flag] < 4){
                    tmp = a.vs[flag];
                    a.vs[flag] = a.vs[flag] * a.vs[flag] - b.vs[flag] * b.vs[flag] + x.vs[flag];
                    b.vs[flag] = 2 * tmp * b.vs[flag] + y.vs[flag];
                    length.vs[flag] = a.vs[flag] * a.vs[flag] + b.vs[flag] * b.vs[flag];
                    ++repeats[flag];
                }
                local_image[(i + flag) * width + local_col] = repeats[flag];

            }   

            if(row_per_proc & 1){
                xs[0] = local_col * interval_x + left;
                repeats[0] = 0;
                local_row = rank + (row_per_proc - 1) * size;
                ys[0] = local_row * interval_y + lower;
                as[0] = bs[0] = 0;

                while(repeats[0] < iters && as[0] * as[0] + bs[0] * bs[0] < 4){
                    tmp = as[0];
                    as[0] = as[0] * as[0] - bs[0] * bs[0] + xs[0];
                    bs[0] = 2 * tmp * bs[0] + ys[0];
                    ++repeats[0];
                }
                local_image[(row_per_proc - 1) * width + local_col] = repeats[0];
            }
        }
    }
    
    
    MPI_Gatherv(local_image, rcount[rank], MPI_INT, image, rcount, displs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();

    if(rank == 0){
        write_png(filename, iters, width, height, image);
        free(image);
    }

}
