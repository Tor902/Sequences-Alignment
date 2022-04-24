#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <limits.h>
#include "cudaHeader.h"

__device__ __constant__ char hashA[ALPHABET][ITEM_MAX_LEN] = {"ST","","","NEQ","NDQK","YWMIL","","QRKYN","MLFV","","NEQHR","MIVF","ILVF","DEQHK","","","NDEHRK","QHK","TA","SA","","MIL","FY","","FWH",""};
__device__ __constant__ char hashB[ALPHABET][ITEM_MAX_LEN] = {"SGTVCP","","SA","SGNEQHK","NQHRKDS","HYVLIM","SAND","NEQRKDFY","FVLM","","NEQHRDST","FVIM","FVLI","SGDEQHRKT","","STA","NEHRKDS","NEQHK","AGNDTPCKEQ","AVSPNK","","TAFLIM","","","HF",""};


__device__ int check_group_similarity_cuda(char hashX[ALPHABET][ITEM_MAX_LEN], char seq1_char, char seq2_char)
{	

	int i;
	int letter_index = seq1_char - 'A';
	if(hashX[letter_index] == "")
		return 0;

	for(i =0 ; i<ITEM_MAX_LEN ;i++)
	{
		if(hashX[letter_index][i] == seq2_char)
			return 1;
	}
	return 0;
}


__global__ void thread_score_offset_calc(char* seq1, char* seq2 , int* lenght_seq2 , int* weights, int* start, int* end, int* res)
{
    int total_mutations = *lenght_seq2;
    int score = 0;
    int index1 = 0;

    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int mutant =  thread_id;
    int thread_index_for_res_array = thread_id * RES_CACHE_LEN; 

    if(mutant <= total_mutations && mutant !=0)
    {
        for (int offset = *start; offset < *end; offset++) 
        {
            index1 = offset;
            for (int index2 = 0; index2 < *lenght_seq2; index2++)
            {
                if (index2 == mutant)
                    index1++;
                
                if (seq1[index1] == seq2[index2])
                    score += weights[DOLLAR];
                
                else if (check_group_similarity_cuda(hashA, seq1[index1], seq2[index2]))
                    score -= weights[PRECENT];
                
                else if (check_group_similarity_cuda(hashB, seq1[index1], seq2[index2]))
                    score -= weights[HASH];
                
                else
                    score -= weights[SPACE];
                index1++;
            }
            
            if(score > res[thread_index_for_res_array])
            {
                res[thread_index_for_res_array] = score;
                res[thread_index_for_res_array + OFFSET] = offset;
                res[thread_index_for_res_array + MUTANT] = mutant;
            }
            score = 0;
        }
    } 
}

//CHECK FUNCTION
void checkStatus(cudaError_t cudaStatus, std::string err)
{
    if(cudaStatus != cudaSuccess)
    {
        std::cout << err << std::endl;
        exit(1);

    }
}


int* cuda_score_offset_mutant(char* seq1, char* seq2 , int* weights, int offset_start, int offset_end)
{
    cudaError_t cudaStatus;

    // EACH THREAD CHECK ALL OFFSETS FOR A SINGLE MUTATION
    int* optimal_res = (int*)malloc(sizeof(int)*RES_CACHE_LEN);

    int seq2_lenght = (strlen(seq2));
    int total_mutations = seq2_lenght;
    int num_of_blocks = (total_mutations / THREADS_PER_BLOCK);
    if (total_mutations % THREADS_PER_BLOCK != 0)
        num_of_blocks ++;
    
    // POINTERS FOR CUDA MEM
    char* cuda_seq1, *cuda_seq2;
    int* cuda_weights, *cuda_offset_start, *cuda_offset_end ,*cuda_seq2_len;
    int* cuda_res_array, *res_array;

    int res_array_size = total_mutations * (RES_CACHE_LEN);
    res_array = (int*)malloc(sizeof(int)*res_array_size);
    for (int i = 0; i < res_array_size; i +=3)
        res_array[i] = INT_MIN;
    

    // MEM ALLOC IN CUDA
    int cuda_seq1_size = sizeof(char) * (strlen(seq1));
    int cuda_seq2_size = sizeof(char) * seq2_lenght;
    int cuda_weights_arr_size = sizeof(int) * (WEIGHTS);

    cudaStatus = cudaMalloc((void**)&cuda_seq1, cuda_seq1_size);
    checkStatus(cudaStatus,"cudaMalloc Failed\n");
    cudaStatus = cudaMalloc((void**)&cuda_seq2_len, sizeof(int));
    checkStatus(cudaStatus,"cudaMalloc Failed\n");
    cudaStatus = cudaMalloc((void**)&cuda_seq2, cuda_seq2_size);
    checkStatus(cudaStatus,"cudaMalloc Failed\n");
    cudaStatus = cudaMalloc((void**)&cuda_weights, cuda_weights_arr_size);
    checkStatus(cudaStatus,"cudaMalloc Failed\n");
    cudaStatus = cudaMalloc((void**)&cuda_offset_start, sizeof(int));
    checkStatus(cudaStatus,"cudaMalloc Failed\n");
    cudaStatus = cudaMalloc((void**)&cuda_offset_end, sizeof(int));
    checkStatus(cudaStatus,"cudaMalloc Failed\n");
    cudaStatus = cudaMalloc((void**)&cuda_res_array, sizeof(int) * res_array_size);
    checkStatus(cudaStatus,"cudaMalloc Failed\n");


    // SEND DATA TO GPU
    cudaStatus = cudaMemcpy(cuda_seq1, seq1, cuda_seq1_size, cudaMemcpyHostToDevice);
    checkStatus(cudaStatus,"cudaMemcpy Failed\n");
    cudaStatus = cudaMemcpy(cuda_seq2_len, &seq2_lenght, sizeof(int), cudaMemcpyHostToDevice);
    checkStatus(cudaStatus,"cudaMemcpy Failed\n");
    cudaStatus = cudaMemcpy(cuda_seq2, seq2, cuda_seq2_size, cudaMemcpyHostToDevice);
    checkStatus(cudaStatus,"cudaMemcpy Failed\n");
    cudaStatus = cudaMemcpy(cuda_weights, weights, cuda_weights_arr_size, cudaMemcpyHostToDevice);
    checkStatus(cudaStatus,"cudaMemcpy Failed\n");
    cudaStatus = cudaMemcpy(cuda_offset_start, &offset_start, sizeof(int), cudaMemcpyHostToDevice);
    checkStatus(cudaStatus,"cudaMemcpy Failed\n");
    cudaStatus = cudaMemcpy(cuda_offset_end, &offset_end, sizeof(int), cudaMemcpyHostToDevice);
    checkStatus(cudaStatus,"cudaMemcpy Failed\n");
    cudaStatus = cudaMemcpy(cuda_res_array, res_array, sizeof(int) * res_array_size, cudaMemcpyHostToDevice);
    checkStatus(cudaStatus,"cudaMemcpy Failed\n");
    
    // LUNCH GPU
    thread_score_offset_calc<<<num_of_blocks, THREADS_PER_BLOCK>>>(cuda_seq1, cuda_seq2, cuda_seq2_len, cuda_weights, cuda_offset_start, cuda_offset_end ,cuda_res_array);

    cudaStatus = cudaDeviceSynchronize();

    // RECV ALL RESULTS - 1 PER EACH THREAD
    cudaStatus = cudaMemcpy(res_array, cuda_res_array, sizeof(int)*res_array_size, cudaMemcpyDeviceToHost);
    checkStatus(cudaStatus,"cudaMemcpy Failed\n");
    // GET THE BIGGEST SCORE
    optimal_res[SCORE] = INT_MIN; 

    for (int i = 0; i < res_array_size; i += 3)
    {
        if(optimal_res[SCORE] < res_array[i])
        {
            optimal_res[SCORE] = res_array[i];
            optimal_res[OFFSET] = res_array[i+OFFSET];
            optimal_res[MUTANT] = res_array[i+MUTANT];
        }
    }
    
    // FREE 
    cudaFree(cuda_seq1);
    cudaFree(cuda_seq2);
    cudaFree(cuda_weights);
    cudaFree(cuda_offset_start);
    cudaFree(cuda_offset_end);
    cudaFree(cuda_res_array);
    cudaFree(cuda_seq2_len);

    return optimal_res;
}
