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
__device__ volatile int g_mutex = 0;
// __device__ volatile int g_mutex1 = 0;
// __device__ volatile int g_mutex2 = 0;
// __device__ volatile int g_mutex3 = 0;
// __device__ volatile int g_mutex4 = 0;

const char *MyGetRuntimeError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        return cudaGetErrorString(error);
    }
    else
        return NULL;
}

char *MyGetdeviceError(CUresult error)
{
    if (error != CUDA_SUCCESS)
    {
        char *charerr = (char *)malloc(100);
        cuGetErrorString(error, (const char **)&charerr);
        return charerr;
    }
    else
        return NULL;
}

// GPU lock-based synchronization function
__device__ void __gpu_sync(int times)
{
    // thread ID in a block
    int goalVal = 2;
    int tid_in_block = getThreadIdInBlock();

    if (tid_in_block == 0)
    {
        atomicAdd((int *)&g_mutex, 1);
        printf("Block %d 's mutex is %d , kernel syn goal: %d.\n", getBlockIDInGrid(), g_mutex, times);
        while (g_mutex != goalVal)
        {
            printf("");
            // Do nothing here. Until for synchronization
        }

        //到达屏障后
        atomicAdd((int *)&g_mutex, -1);
        printf("Syn over, Block %d 's mutex--, now is %d.\n", getBlockIDInGrid(), g_mutex);
        while (g_mutex != 0)
        {
            printf("");
            // Do nothing here. Until mutex to 0
        }
    }
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

__global__ void cache(int clockRate, DATATYPE *GPU_array_L1, DATATYPE *GPU_array_L2, DATATYPE *dura, int kernelID)
{
    // int array_num = L1_SIZE / sizeof(DATATYPE) / strige + 1;
    uint32_t i = 0;
    uint32_t step = 0;
    uint32_t time = 0;
    // shared memory size : 24KB
    const uint32_t SM_size = 24 * 1024 / sizeof(DATATYPE);
    __shared__ DATATYPE s_tvalue[SM_size];

    uint32_t smid = getSMID();
    uint32_t blockid = getBlockIDInGrid();
    uint32_t threadid = getThreadIdInBlock();

    printf("Here is kernel %d Blcok %d , running in sm %d.\n", kernelID, blockid, smid);

    bool kL1hit = false;
    bool kL2hit = false;
    //程序0运行在sm0上的block才运行
    if (kernelID == 0 && smid == 0)
    {
        kL1hit = true;
    }
    //程序1运行在sm2上的block才运行
    if (kernelID == 1 && smid == 2)
    {
        kL2hit = true;
    }
    __syncthreads();
    if (kL1hit || kL2hit)
    {
        // L1 hit
        i = threadid;
        for (int j = 0; j < 5; j++)
            while (i < L1_limit)
            {
                i = GPU_array_L1[i];
                step++;
                // if (threadid == 0 && blockid == 0)
                // printf("Thread : %d \t step : %d \t i : %d \t Limit is %d\n", threadid, step, i, L1_limit);
            }
        // printf("step is : %d\n", step);

        printf("Kernel %d 's block %d loading L1 cache in sm %d over.\n", kernelID, blockid, smid);
        // __gpu_sync(2);

        // hit L1 cache function
        auto hit_L1 = [&](int count)
        {
            for (int j = 0; j < count; j++)
            {
                uint32_t index = 1;
                i = 0;
                DATATYPE Start_time = get_time(clockRate);
                while (i < L1_limit)
                {
                    i = GPU_array_L1[i];
                    ++index;
                    DATATYPE End_time = get_time(clockRate);
                    s_tvalue[index + (step * time)] = End_time - Start_time;
                    // if ((index + (step * time)) % 32 == 0)
                    //     printf("%d——%d testing L1, %d duration is %.4f\n", (time + 2) / 2, time + 1, index + (step * time), s_tvalue[index + (step * time)]);
                }

                printf("\nKernel %d 's block %d in sm %d || %d——%d testing L1 over, %d duration is %.4f\n", kernelID, blockid, smid, (time + count) / count, time % count + 1, index + (step * time), s_tvalue[index + (step * time)]);
                ++time;
                __syncthreads();
            }
        };
        time = 0;
        printf("\nHere is kernel %d Blcok %d , running in sm %d.\n", kernelID, blockid, smid);
        // Load L1 cache
        if (kL1hit)
        {
            DATATYPE L1time = get_time(clockRate);
            // 1-1 test L1
            hit_L1(3);
            float durationL1 = get_time(clockRate) - L1time;
            printf("Here is kernel %d Blcok %d , hit_L1(3) duration is %.6f.\n", kernelID, blockid, durationL1);
        }
        // __syncthreads();
        // if (threadid == 0)
        printf("start sys 2.\n");
        //等待L1 hit完毕
        // fence[0] += blockid * threadid;
        // __threadfence();
        // __gpu_sync(2);
        printf("I'm Kernel %d 's Block %d in sm %d. Note: Start hit L2.\n", kernelID, blockid, smid);

        // Load L2 cache
        if (kL2hit)
        {
            DATATYPE L2time = get_time(clockRate);
            for (i = threadid; i < L2_SIZE;)
            {
                i = GPU_array_L2[i];
            }
            float durationL2 = get_time(clockRate) - L2time;
            printf("\n Kernel %d 's Block %d in sm %d || Loading data into L2 cost %.6f.\n", kernelID, blockid, smid, durationL2);
        }

        printf("start sys 3.\n");
        __gpu_sync(2);
        printf("I'm Kernel %d 's Block %d in sm %d. Note: Start hit L1 again. \n", kernelID, blockid, smid);
        // Load L1 cache again
        if (kL1hit)
        {
            hit_L1(3);
            //保存访问时间
            s_tvalue[0] = step;
            s_tvalue[1] = time;
            for (i = 0; i < step * time + 2; i++)
            {
                dura[i] = s_tvalue[i];
            }
            //

            printf("\nKernel %d's Block %d in sm %d test cache over. step : %.0f, Total times: %.0f\n", kernelID, blockid, smid, dura[0], dura[1]);
        }
        // __syncthreads();
        __gpu_sync(2);
    }
    //等待L1 load again完毕
    // fence[1] += blockid * threadid;
    // __threadfence();
    __syncthreads();
}

void main_test(int clockRate, DATATYPE *array_L1, DATATYPE *array_L2, DATATYPE *dura, int *device)
{
    const int CONTEXT_POOL_SIZE = 2;
    // const int      CONTEXT_POOL_SIZE = 4;
    CUcontext contextPool[CONTEXT_POOL_SIZE];
    int smCounts[CONTEXT_POOL_SIZE];
    int numBlocks[CONTEXT_POOL_SIZE];
    for (int i = 0; i < CONTEXT_POOL_SIZE; i++)
    {
        smCounts[i] = 4;

        CUexecAffinityParam affinity;
        affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
        affinity.param.smCount.val = smCounts[i];

        CUresult err;
        err = cuCtxCreate_v3(&contextPool[i], &affinity, 1, 0, *device);

        if (MyGetdeviceError(err) != NULL)
        {
            printf("The %d cuCtxCreate_v3 Error:%s\n", i, MyGetdeviceError(err));
        }
        // cuCtxCreate_v3 创建带有affinity的上下文，并且CU_EXEC_AFFINITY_TYPE_SM_COUNT属性仅在Volta及更新的架构上以及MPS下可用
        //链接：https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g2a5b565b1fb067f319c98787ddfa4016
        // cuCtxCreate_v3(&contextPool[i], &affinity, 1, 0, deviceOrdinal);
    }

    int blocks = 4;
    int threads = 1;
    // int dura_num = 5;

    DATATYPE *GPU_array_L1;
    DATATYPE *GPU_array_L2;
    DATATYPE *GPU_dura;
    cudaMalloc((void **)&GPU_array_L1, L1_SIZE);
    cudaMalloc((void **)&GPU_array_L2, sizeof(DATATYPE) * L2_SIZE);
    cudaMalloc((void **)&GPU_dura, SHARED_SIZE);
    cudaMemcpy(GPU_array_L1, array_L1, L1_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_array_L2, array_L2, sizeof(DATATYPE) * L2_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_dura, dura, SHARED_SIZE, cudaMemcpyHostToDevice);

    // printf("init shared memory size over.\n");
    std::thread mythread[CONTEXT_POOL_SIZE];
    for (int kernelID = 0; kernelID < CONTEXT_POOL_SIZE; kernelID++)
        mythread[kernelID] = std::thread([&, kernelID]()
                                         {
            int                 numSms = 0;
            CUexecAffinityParam affinity;

            CUresult err1;
            //将指定的CUDA上下文绑定到调用CPU线程
            err1 = cuCtxSetCurrent(contextPool[kernelID]);
            if (err1 != CUDA_SUCCESS) {
                printf("thread cuCtxSetCurrent Error:%s\n", MyGetdeviceError(err1));
            }

            CUresult err2;
            // Returns the execution affinity setting for the current context
            err2 = cuCtxGetExecAffinity(&affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
            if (err2 != CUDA_SUCCESS) {
                printf("thread cuCtxGetExecAffinity Error:%s\n", MyGetdeviceError(err2));
            }

            //获取当前context对应的线程数目
            numSms = affinity.param.smCount.val;
            if (numSms != smCounts[kernelID]) {
                printf("Context %d parititioning SM error!\tPlan:%d\tactual:%d\n", kernelID, smCounts[kernelID], numSms);
                // cout<< "Context "<< step << " parititioning SM error!\tPlan:" <<
                // smCounts[step] << "\tactual:" << numSms << endl;
            } else {
                printf("Context %d parititioning SM success!\tPlan:%d\tactual:%d\n", kernelID, smCounts[kernelID], numSms);
            }
            // kernel here
            cache<<<blocks, threads>>>(clockRate, GPU_array_L1, GPU_array_L2, GPU_dura,kernelID);
            cudaDeviceSynchronize(); });
    for (int kernelID = 0; kernelID < CONTEXT_POOL_SIZE; kernelID++)
        mythread[kernelID].join();
    cudaDeviceReset();

    cudaMemcpy(dura, GPU_dura, SHARED_SIZE, cudaMemcpyDeviceToHost);
    //读写文件。文件存在则被截断为零长度，不存在则创建一个新文件
    FILE *fp = fopen("./out/cache.csv", "w+");
    if (fp == NULL)
    {
        fprintf(stderr, "fopen() failed.\n");
        exit(EXIT_FAILURE);
    }

    int step = dura[0];
    int time = dura[1];

    fprintf(fp, "step,");
    for (int i = 1; i <= 2; i++)
    {
        for (int j = 1; j <= time / 2; j++)
        {
            fprintf(fp, "%d-%dhit,", i, j);
        }
    }
    fprintf(fp, "\n");

    for (int i = 0; i < step; i++)
    {
        fprintf(fp, "%d,", i + 1);
        for (int j = 0; j < time; j++)
        {
            int index = i + 2 + step * j;
            fprintf(fp, "%.4f,", dura[index]);
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
    DATATYPE *dura;
    array_L1 = (DATATYPE *)malloc(L1_SIZE);
    array_L2 = (DATATYPE *)malloc(sizeof(DATATYPE) * L2_SIZE);
    dura = (DATATYPE *)malloc(SHARED_SIZE);
    init_order(array_L1, L1_limit, flag);
    init_order(array_L2, L2_SIZE, flag);
    init_order(dura, SHARED_SIZE / sizeof(DATATYPE), 0);

    main_test(clockRate, array_L1, array_L2, dura, &device);

    free(array_L1);
    free(array_L2);
    free(dura);

    printf("\nAll done.\n");
    return 0;
}