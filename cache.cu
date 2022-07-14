#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <cuda.h>
#include "util.cu"
#include <iostream>
#include <utility>
#include <thread>

using namespace std;

#define DATATYPE float
#define L1_MAX_SIZE 131072
#define L1_SIZE 64 * 1024
#define SHARED_SIZE 64 * 1024
#define L2_SIZE 2359296
#define strige 16 / sizeof(DATATYPE)

//初始化数组，a[i]=0
template <class T>
void init_order(T *a, int n, int flag)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = ((i + stride) % n) * flag;
    }
}

DATATYPE get_time(int clockRate)
{
    clock_t now_clock = clock();
    return (DATATYPE)now_clock / clockRate;
}

__global__ void cache(int clockRate, DATATYPE *GPU_array_L1, DATATYPE *GPU_array_L2, DATATYPE **dura)
{
    int array_num = L1_SIZE / sizeof(DATATYPE) / strige + 1;
    int i = 0;
    int step = 0;
    __shared__ DATATYPE s_tvalue[array_num];
    extern __shared__ DATATYPE s2_tvalue[];
    // __shared__ DATATYPE s_index[array_num];

    uint32_t smid = getSMID();
    uint32_t blockid = getBlockIDInGrid();
    uint32_t threadid = getThreadIdInBlock();
    printf("Blcok %d is running in sm %d.\n", blockid, smid);

    // L1 hit
    while (i < L1_SIZE / sizeof(DATATYPE))
    {
        i = GPU_array_L1[i];
        step++;
    }

    // Load L1 cache
    if (blockid == 0)
    {
        step = 0;
        int index = 0;
        for (i = 0; i < L1_SIZE / sizeof(DATATYPE);)
        {
            DATATYPE Start_time = get_time(clockRate);
            i = GPU_array_L1[i];
            step++;
            DATATYPE End_time = get_time(clockRate);
            s_tvalue[index++] = End_time - Start_time;
            printf("First testing L1, %d duration is %.4f\n", step, End_time - Start_time);
        }
    }

    //等待L1 hit完毕
    cudaDeviceSynchronize();

    // Load L2 cache
    if (blockid != 0)
    {
        for (i = 0; i < L2_SIZE;)
        {

            i = GPU_array_L2[i];
        }
    }
    else
        printf("Loading data into L2 cache...\n");

    //等待L2 load完毕
    cudaDeviceSynchronize();

    // Load L1 cache again
    if (blockid == 0)
    {
        step = 0;
        int index = 0;
        for (i = 0; i < L1_SIZE / sizeof(DATATYPE);)
        {
            DATATYPE Start_time = get_time(clockRate);
            i = GPU_array_L1[i];
            step++;
            DATATYPE End_time = get_time(clockRate);
            s2_tvalue[index++] = End_time - Start_time;
            printf("Second testing L1, %d duration is %.4f\n", step, End_time - Start_time);
        }

        //保存两次的访问时间
        for (index = 0; index < step; index++)
        {
            dura[0][index] = s_tvalue[index];
            dura[1][index] = s2_tvalue[index];
        }
        dura[2][0] = step;
    }
    else
        printf("Loading data into L1 cache again...\n");

    //等待L1 load again完毕
    cudaDeviceSynchronize();
}

void main_test(int clockRate, DATATYPE *array_L1, DATATYPE *array_L2)
{
    int blocks = 2;
    int threads = 1;
    int dura_num = 3;
    DATATYPE **dura;
    dura = (DATATYPE **)malloc(sizeof(DATATYPE *) * dura_num);
    for (int i = 0; i < dura_num; i++)
    {
        //初始化为0
        dura[i] = (DATATYPE *)malloc(L1_SIZE);
        init_order(dura[i], L1_SIZE / sizeof(DATATYPE), 0);
    }

    DATATYPE *GPU_array_L1;
    DATATYPE *GPU_array_L2;
    cudaMalloc((void **)&GPU_array_L1, L1_SIZE);
    cudaMalloc((void **)&GPU_array_L2, sizeof(DATATYPE) * L2_SIZE);
    cudaMemcpy(GPU_array_L1, array_L1, L1_SIZE);
    cudaMemcpy(GPU_array_L2, array_L2, sizeof(DATATYPE) * L2_SIZE);

    cudaFuncSetAttribute(cache, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_SIZE);

    // kernel here
    cache<<<blocks, threads, 32 * 1024>>>(clockRate, GPU_array_L1, GPU_array_L2, dura);

    cudaDeviceSynchronize();

    //读写文件。文件存在则被截断为零长度，不存在则创建一个新文件
    FILE *fp = fopen("./out/cache.csv", "w+");
    if (fp == NULL)
    {
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "step,1_L1_duration,2_L1_duration\n");
    for (int i = 0; i < dura[2][0]; i++)
    {
        fprintf(fp, "%d,%.4f,%.4f\n",i,dura[0][i],dura[1][i]);
    }
    
    fclose(fp);

    cudaFree(GPU_array_L1);
    cudaFree(GPU_array_L2);
    for (int i = 0; i < dura_num; i++)
    {
        free(dura[i]);
    }
    free(dura);
}

int main()
{
    int device = 0;
    int flag = 1;
    cudaDeviceProp prop;
    cudaSetDevice(device);
    // printf("device:%d\n",device);
    cudaGetDeviceProperties(&prop, device);
    int clockRate = prop.clockRate;
    int sm_number = prop.multiProcessorCount;
    printf("*********   This GPU has %d SMs   *********\n", sm_number);
    // output GPU prop

    DATATYPE *array_L1;
    DATATYPE *array_L2;
    array_L1 = (DATATYPE *)malloc(L1_SIZE);
    array_L1 = (DATATYPE *)malloc(sizeof(DATATYPE) * L2_SIZE);
    init_order(array_L1, L1_SIZE / sizeof(DATATYPE), flag);
    init_order(array_L2, L2_SIZE, flag);

    main_test(clockRate, array_L1, array_L2);

    free(array_L1);
    free(array_L2);

    return 0;
}