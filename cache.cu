#include "util.cu"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cuda.h>
#include <iostream>
#include <utility>
#include <thread>

using namespace std;

#define DATATYPE float
#define L1_MAX_SIZE 131072
// 64KB 64 * 1024 = 65536
#define L1_SIZE 65536
// 64KB 64 * 1024 = 65536
#define SHARED_SIZE 65536
#define L2_SIZE 2359296
#define strige 8
// L1_SIZE / sizeof(DATATYPE) = 16384
#define L1_limit 16384
// #define factor 0.8
// lock-based
__device__ volatile int g_mutex1 = 0;
__device__ volatile int g_mutex2 = 0;
__device__ volatile int g_mutex3 = 0;
__device__ volatile int g_mutex4 = 0;

// GPU lock-based synchronization function
__device__ void __gpu_sync(int times)
{
    // thread ID in a block
    int goalVal = 2;
    int tid_in_block = getThreadIdInBlock();

    // auto wait = [goalVal, times](int g_mutex)
    // {
    //     printf("Block %d 's mutex is %d , Times: %d .\n", getBlockIDInGrid(), g_mutex, times);
    //     // only when all blocks add 1 go g_mutex
    //     // will g_mutex equal to goalVal
    //     while (g_mutex != goalVal)
    //     {
    //         // Do nothing here. Until for synchronization
    //     }
    // };
    // only thread 0 is used for synchronization
    if (tid_in_block == 0)
    {
        switch (times)
        {
        case 1:
            atomicAdd((int *)&g_mutex1, 1);
            printf("Block %d 's mutex is %d , kernel syn times: %d.\n", getBlockIDInGrid(), g_mutex1, times);
            // only when all blocks add 1 go g_mutex
            // will g_mutex equal to goalVal
            while (g_mutex1 != goalVal)
            {
                printf("");
                // Do nothing here. Until for synchronization
            }
            break;
        case 2:
            atomicAdd((int *)&g_mutex2, 1);
            printf("Block %d 's mutex is %d , kernel syn times: %d.\n", getBlockIDInGrid(), g_mutex2, times);
            // only when all blocks add 1 go g_mutex
            // will g_mutex equal to goalVal
            while (g_mutex2 != goalVal)
            {
                printf("");
                // Do nothing here. Until for synchronization
            }
            break;
        case 3:
            atomicAdd((int *)&g_mutex3, 1);
            printf("Block %d 's mutex is %d , kernel syn times: %d.\n", getBlockIDInGrid(), g_mutex3, times);
            // only when all blocks add 1 go g_mutex
            // will g_mutex equal to goalVal
            while (g_mutex3 != goalVal)
            {
                printf("");
                // Do nothing here. Until for synchronization
            }
            break;
        case 4:
            atomicAdd((int *)&g_mutex4, 1);
            printf("Block %d 's mutex is %d , kernel syn times: %d.\n", getBlockIDInGrid(), g_mutex4, times);
            // only when all blocks add 1 go g_mutex
            // will g_mutex equal to goalVal
            while (g_mutex4 != goalVal)
            {
                printf("");
                // Do nothing here. Until for synchronization
            }
            break;

        default:
            printf("Error sys times: %d .  Exit.\n", times);
            break;
        }
    }
    __syncthreads();
}

//初始化数组，a[i]=0
template <class T>
void init_order(T *a, int n, int flag)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = (i + strige) * flag;
    }
}

__global__ void cache(int clockRate, DATATYPE *GPU_array_L1, DATATYPE *GPU_array_L2, DATATYPE **dura)
{
    // int array_num = L1_SIZE / sizeof(DATATYPE) / strige + 1;
    uint32_t i = 0;
    uint32_t step = 0;
    // 16385 / 8 + 1 = 2048+ 1
    __shared__ DATATYPE s_tvalue[L1_limit / strige + 1];
    __shared__ DATATYPE s1_tvalue[L1_limit / strige + 1];
    __shared__ DATATYPE s2_1_tvalue[L1_limit / strige + 1];
    extern __shared__ DATATYPE s2_tvalue[];
    // __shared__ DATATYPE fence[2];

    uint32_t smid = getSMID();
    uint32_t blockid = getBlockIDInGrid();
    uint32_t threadid = getThreadIdInBlock();
    __syncthreads();
    printf("Blcok %d is running in sm %d.\n", blockid, smid);

    // L1 hit
    i = threadid;
    while (i < L1_limit)
    {
        i = GPU_array_L1[i];
        step++;
        // if (threadid == 0 && blockid == 0)
        // printf("Thread : %d \t step : %d \t i : %d \t Limit is %d\n", threadid, step, i, L1_limit);
    }

    __gpu_sync(1);
    if (threadid == 0)
        printf("block %d test loading L1 cache over.\n", blockid);

    // Load L1 cache
    if (blockid == 0)
    {
        // 1-1 test L1
        step = 0;
        i = threadid;
        DATATYPE Start_time = get_time(clockRate);
        while (i < L1_limit)
        {
            i = GPU_array_L1[i];
            step++;
            DATATYPE End_time = get_time(clockRate);
            s_tvalue[step - 1] = End_time - Start_time;
            if (step % 32 == 0)
                printf("1——1 testing L1, %d duration is %.4f\n", step - 1, s_tvalue[step - 1]);
        }

        printf("1——1 loading over , step is : %d\n", step);

        // 1-2 test L1
        step = 0;
        i = threadid;
        Start_time = get_time(clockRate);
        while (i < L1_limit)
        {

            i = GPU_array_L1[i];
            step++;
            DATATYPE End_time = get_time(clockRate);
            s1_tvalue[step - 1] = End_time - Start_time;
            if (step % 32 == 0)
                printf("1——2 testing L1, %d duration is %.4f\n", step - 1, s1_tvalue[step - 1]);
        }

        printf("1——2 loading over , step is : %d\n", step);
    }
    // __syncthreads();
    // if (threadid == 0)
    else
        printf("Block 1 is wating 0's first loading data into L1 cache...\n");
    //等待L1 hit完毕
    // fence[0] += blockid * threadid;
    // __threadfence();
    __gpu_sync(2);

    // Load L2 cache
    if (blockid != 0)
    {
        for (i = threadid; i < L2_SIZE;)
        {

            i = GPU_array_L2[i];
        }
        printf("Block %d loading data into L2 cache over.\n", blockid);
    }
    else
        printf("Block 0 is waiting for 1's Loading data into L2 cache...\n");

    __gpu_sync(3);

    // Load L1 cache again
    if (blockid == 0)
    {
        step = 0;
        i = threadid;
        DATATYPE Start_time = get_time(clockRate);
        while (i < L1_limit)
        {

            i = GPU_array_L1[i];
            step++;
            DATATYPE End_time = get_time(clockRate);
            s2_1_tvalue[step - 1] = End_time - Start_time;
            if (step % 32 == 0)
                printf("2-1 testing L1, %d duration is %.4f\n", step - 1, s2_1_tvalue[step - 1]);
        }
        printf("2-1 loading over , step is : %d\n", step);

        // 2-2 L1 test
        step = 0;
        i = threadid;
        Start_time = get_time(clockRate);
        while (i < L1_limit)
        {

            i = GPU_array_L1[i];
            step++;
            DATATYPE End_time = get_time(clockRate);
            s2_tvalue[step - 1] = End_time - Start_time;
            if (step % 32 == 0)
                printf("2-2 testing L1, %d duration is %.4f\n", step - 1, s2_tvalue[step - 1]);
        }
        printf("2-2 loading over , step is : %d\n", step);
        // __syncthreads();
        //保存4次的访问时间
        for (i = 0; i < step; i++)
        {
            dura[0][i] = s_tvalue[i];
            dura[1][i] = s1_tvalue[i];
            dura[2][i] = s2_1_tvalue[i];
            dura[3][i] = s2_tvalue[i];
            if (i % 32 == 0)
                printf("duration i : %d, step is : %d \n", i, step);
        }
        // if (threadid == 0)
        dura[4][0] = step;
        printf("Duration 2 over. dura[4][0] : %.0f\n", dura[4][0]);
    }
    // __syncthreads();
    __gpu_sync(4);

    //等待L1 load again完毕
    // fence[1] += blockid * threadid;
    // __threadfence();
}

void main_test(int clockRate, DATATYPE *array_L1, DATATYPE *array_L2,DATATYPE **dura,int dura_num)
{
    int blocks = 2;
    int threads = 1;
    // int dura_num = 5;
    
    DATATYPE *GPU_array_L1;
    DATATYPE *GPU_array_L2;
    cudaMalloc((void **)&GPU_array_L1, L1_SIZE);
    cudaMalloc((void **)&GPU_array_L2, sizeof(DATATYPE) * L2_SIZE);
    cudaMemcpy(GPU_array_L1, array_L1, L1_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_array_L2, array_L2, sizeof(DATATYPE) * L2_SIZE, cudaMemcpyHostToDevice);
    cudaFuncSetAttribute(cache, cudaFuncAttributeMaxDynamicSharedMemorySize, SHARED_SIZE);
    printf("init shared memory size over.\n");
    // kernel here
    cache<<<blocks, threads, 16 * 1024>>>(clockRate, GPU_array_L1, GPU_array_L2, dura);

    cudaDeviceSynchronize();

    //读写文件。文件存在则被截断为零长度，不存在则创建一个新文件
    FILE *fp = fopen("./out/cache.csv", "w+");
    if (fp == NULL)
    {
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(fp, "step,1_1_L1,1_2_L1,2_1_L1,2_2_L1\n");
    for (int i = 0; i < dura[dura_num - 1][0] * threads; i++)
    {
        fprintf(fp, "%d", i);
        for (int j = 0; j < dura_num - 1; j++)
        {
            fprintf(fp, "%.4f", dura[j][i]);
        }

        fprintf(fp, "\n");
    }

    fclose(fp);

    cudaFree(GPU_array_L1);
    cudaFree(GPU_array_L2);
    
}

int main()
{
    int device = 0;
    int flag = 1;
    int dura_num = 5;
    cudaDeviceProp prop;
    cudaSetDevice(device);
    // printf("device:%d\n",device);
    cudaGetDeviceProperties(&prop, device);
    int clockRate = prop.clockRate;
    int sm_number = prop.multiProcessorCount;
    printf("*********   This GPU has %d SMs   *********\n", sm_number);
    // output GPU prop

    printf("L1size: %lu \t sizeoftype:%lu \t L1limt:%lu \t L2size:%lu \n", L1_SIZE, sizeof(DATATYPE), L1_limit, L2_SIZE);
    // getchar();
    DATATYPE *array_L1;
    DATATYPE *array_L2;
    array_L1 = (DATATYPE *)malloc(L1_SIZE);
    array_L2 = (DATATYPE *)malloc(sizeof(DATATYPE) * L2_SIZE);
    init_order(array_L1, L1_limit, flag);
    init_order(array_L2, L2_SIZE, flag);

    DATATYPE **dura;
    dura = (DATATYPE **)malloc(sizeof(DATATYPE *) * dura_num);
    for (int i = 0; i < dura_num; i++)
    {
        //初始化为0
        dura[i] = (DATATYPE *)malloc(sizeof(DATATYPE) * (L1_SIZE / strige));
        init_order(dura[i], L1_limit, 0);
    }

    main_test(clockRate, array_L1, array_L2,dura,dura_num);

    free(array_L1);
    free(array_L2);
    for (int i = 0; i < dura_num; i++)
    {
        free(dura[i]);
    }
    free(dura);
    return 0;
}