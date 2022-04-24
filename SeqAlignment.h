
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <math.h>
#include <ctype.h>
#include <omp.h>
#include "mpi.h"

#define FILE_NAME "input.txt"

#define ROOT 0
#define MAX_LEN_SEQ1 3000
#define MAX_LEN_SEQ2 2000
#define RES_CACHE_LEN 3 //The Result variable composed out of 3 values: score, offset, mutant.

// Groups constants:
#define GROUP_A_ITEMS 9
#define GROUP_B_ITEMS 11
#define ITEM_MAX_LEN 11
#define ALPHABET 26

#define WEIGHTS 4
#define SCORE 0
#define OFFSET 1
#define MUTANT 2
/*
	Simillarity:
	Two chars are Similar under GroupX if they both apear in the same subgroup (item) in GroupX
	or Globaly Similar if they are Equal
*/
#define DOLLAR 0 // Two chars are Globaly Similar (equal)
#define PRECENT 1 // Two chars similar under GroupA
#define HASH 2 // Two chars similar under GroupB
#define SPACE 3 // Two chars NOT similar

enum tags{WORK,TERMINATE};

// Define Groups
char groupA[GROUP_A_ITEMS][ITEM_MAX_LEN] = {"NDEQ", "MILV", "FYW", "NEQK", "QHRK", "HY", "STA", "NHQK", "MILF"};
char groupB[GROUP_B_ITEMS][ITEM_MAX_LEN] = {"SAG", "ATV", "CSA", "SGND", "STPA", "STNK", "NEQHRK", "NDEQHK", "SNDEQK", "HFY", "FVLIM"};

char hashA[ALPHABET][ITEM_MAX_LEN] = {"ST","","","NEQ","NDQK","YWMIL","","QRKYN","MLFV","","NEQHR","MIVF","ILVF","DEQHK","","","NDEHRK","QHK","TA","SA","","MIL","FY","","FWH",""};
char hashB[ALPHABET][ITEM_MAX_LEN] = {"SGTVCP","","SA","SGNEQHK","NQHRKDS","HYVLIM","SAND","NEQRKDFY","FVLM","","NEQHRDST","FVIM","FVLI","SGDEQHRKT","","STA","NEHRKDS","NEQHK","AGNDTPCKEQ","AVSPNK","","TAFLIM","","","HF",""};


int** master_process(char* seq1, char** seq2_arr, int seq2_arr_size, int num_processes, int **seq2_optimal_results);
void worker_process(char* seq1, int* weights, int **table);
int* omp_alignment_score_offset_mutant(char* seq1,char* seq2, int* weights, int max_offset, int* optimal_res, int** table);

int** fill_table(int* weights, int** table);
void print_table(int** table);
int check_group_similarity_hash(char hashX[ALPHABET][ITEM_MAX_LEN], char seq1_char, char seq2_char);

int preprocess_and_check_input(char* secondary_sequence, int seq2_index, int seq1_len);
int** mem_allocation_2d(int** pointer, int n, int m);
int conti_mem_allocation_2d(int*** array, int n, int m);
void conti_mem_free_2d(int ***array);
void print_all_optimal_results(char* seq1, int** seq2_optimal_results, int seq2_arr_size);




// int** master_process(char* seq1, char** seq2_arr, int seq2_arr_size, int num_processes, int** seq2_optimal_results);
/*
	Assign one sequence to each process.
	When a process is done computing and returns the optimal score (with optimal offset and mutant),
	save those values in all_sequences_optimal_results.
	Assign said process a new sequence to compute or TERMINATE that process if all work is already assigned.


	return value:
	all_sequences_optimal_results - each result composed of 3 optimal components: score,offset,mutant
*/

// void worker_process(char* seq1, int* weights, int **table);
/*
	Receive work from master process (if received TERMINATE order, free all memory).
	Devide the workload between OpenMP and CUDA.
	Send the optimal Result back to master process.
*/


// int* omp_alignment_score_offset_mutant(char* seq1,char* seq2, int* weights, int max_offset, int* optimal_res, int** table);
/*
	Computing the alignment score of two sequances by checking all possible (offset,mutant) combinations
	IN PARALLEL using openMP


	How The alignment score computes given seq1_char, seq2_char:
	use ONE of the following conditions:

	1. if the charachters are the same, add DOLLAR weight to Result.
	2. if the charachters in the same subgroup in group A|B, reduce PRECENT|HASH weight from Result.
	3. if conditions 1 and 2 not accepted, reduce SPACE weight from Result.

	return value:
	res - optimal values of: score,offset,mutant that gives the Highest score

*/
