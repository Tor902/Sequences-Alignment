#include "cudaHeader.h"
#include "SeqAlignment.h"

// **** Parallel Main ****

int main(int argc, char *argv[])
{
	double timer;
    int my_rank, num_processes;
	int seq2_arr_size, seq1_len;
	int* weights;
	int** seq2_optimal_results;
 	int** table;

    char *seq1, **seq2_arr;

	seq1 = (char*)malloc(sizeof(char)*MAX_LEN_SEQ1);
	weights = (int*)malloc(sizeof(int)*WEIGHTS);
	conti_mem_allocation_2d(&table,ALPHABET, ALPHABET);

    //MPI INIT
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes); 
	

	// ROOT PROCESS INIT
    if(my_rank == ROOT)
    {
	// READ INPUT
		FILE* input = fopen(FILE_NAME, "r");
		if(!input)
			return -1;
		
		// READ WEIGHTS
		fscanf(input, "%d %d %d %d", &weights[DOLLAR], &weights[PRECENT], &weights[HASH], &weights[SPACE]);

		// READ SEQ1
		fscanf(input, "%s", seq1);

		// READ THE AMOUNT OF SEQ2
		fscanf(input, "%d", &seq2_arr_size);
		
		// MEM ALLOC + READ ALL SEQ2
		seq2_arr = (char**)malloc(seq2_arr_size*(sizeof(char*)));
		if(!seq2_arr)
			return -1;
		
		for(int i=0 ; i<seq2_arr_size ; i++)
		{
			seq2_arr[i] = (char*)malloc(MAX_LEN_SEQ2*(sizeof(char)));
			if(!seq2_arr[i])
				return -1;

			fscanf(input, "%s", seq2_arr[i]);
		}
		fclose(input); 


	// CHECK INPUT
		// CHECK VALID SEQ1
		seq1_len = strlen(seq1);
		if(seq1_len > MAX_LEN_SEQ1)
			return -1;

		//CHECK IF AMOUNT OF SEQ2 IS LOWER THAN NUM PROCESSES
        if (seq2_arr_size < num_processes - 1)
        {
            printf("ERROR: THE PROGRAM HAS LESS SEQ2 THAN ACTIVE PROCESSES\n");
            return -1;
        }

		// UPPER CASE AND CHECK SEQ2
		for(int i=0 ; i<seq2_arr_size ; i++)
		{
			if (!preprocess_and_check_input(seq2_arr[i], i, seq1_len))
			{	
				printf("EORROR in preprocess_and_check_input number %d\n",i);
				return -1;
			}
		}

		// MEM ALLOC FOR OPTIMAL RESULTS
		table = fill_table(weights, table);
		seq2_optimal_results = mem_allocation_2d(seq2_optimal_results, seq2_arr_size, RES_CACHE_LEN);
	}
	
	// BROADCAST SEQ1, WEIGHTS AND SCORES TABLE TO ALL PROCESSES
	MPI_Bcast(&(table[0][0]), ALPHABET*ALPHABET, MPI_INT, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(seq1, MAX_LEN_SEQ1, MPI_CHAR, ROOT, MPI_COMM_WORLD);
	MPI_Bcast(weights, WEIGHTS, MPI_INT, ROOT, MPI_COMM_WORLD);
	

	// START
	if(my_rank == ROOT)
	{
		timer = MPI_Wtime();
		seq2_optimal_results = master_process(seq1, seq2_arr, seq2_arr_size, num_processes, seq2_optimal_results);	
		timer = MPI_Wtime() - timer;
	}
	else
	{
		worker_process(seq1, weights, table);
	}
	
	// Final Steps:
	if(my_rank == ROOT)
	{
		// Print results to File + time
		printf("Time to calc Parallel is %lf seconds\n", timer);
		print_all_optimal_results(seq1, seq2_optimal_results, seq2_arr_size);	
	
		// Free Memory
		for (int i = 0; i < seq2_arr_size; i++)
		{
			free(seq2_optimal_results[i]);
			free(seq2_arr[i]);
		}
		free(seq2_optimal_results);
		free(seq2_arr);
	}

	// FREE
	free(seq1);
	free(weights);
	conti_mem_free_2d(&table);
	MPI_Finalize();
	return 0;
}

int** master_process(char* seq1, char** seq2_arr, int seq2_arr_size, int num_processes, int **seq2_optimal_results)
{
	MPI_Status status;
	int tag = WORK;
	int seq2_received=0 ,seq2_sent=0;
	int worker_id;
	int message=0;	
	int *map_worker_2_seq;
	int* buffer_ptr;

	map_worker_2_seq = (int*)malloc(sizeof(int)*num_processes);
	if(!map_worker_2_seq)
		return 0;
	
	
	// SEND ONE SEQ2 TO EACH PROCESS
	for(worker_id=1 ; worker_id<num_processes ; worker_id++)
	{
		MPI_Send(seq2_arr[seq2_sent], MAX_LEN_SEQ2, MPI_CHAR, worker_id, tag, MPI_COMM_WORLD);
		map_worker_2_seq[worker_id] = seq2_sent;
		seq2_sent++;
	}
	
	// RECEIVE AND SEND MORE WORK (UNTIL ALL ARE SENT)
	while(seq2_sent < seq2_arr_size)
	{
		if(seq2_sent >= seq2_arr_size - (num_processes-1))
			tag = TERMINATE;
		
		MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		buffer_ptr = seq2_optimal_results[map_worker_2_seq[status.MPI_SOURCE]];
		MPI_Recv(buffer_ptr, RES_CACHE_LEN, MPI_INT, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		seq2_received++;

		if(status.MPI_TAG != TERMINATE)
		{
			MPI_Send(seq2_arr[seq2_sent], MAX_LEN_SEQ2, MPI_CHAR, status.MPI_SOURCE, tag, MPI_COMM_WORLD);
			map_worker_2_seq[status.MPI_SOURCE] = seq2_sent;
			seq2_sent++;
		}
	}

	while (seq2_received < seq2_arr_size)
	{
		MPI_Recv(&message, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		buffer_ptr = seq2_optimal_results[map_worker_2_seq[status.MPI_SOURCE]];
		MPI_Recv(buffer_ptr, RES_CACHE_LEN, MPI_INT, status.MPI_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		seq2_received++;
	}
	free(map_worker_2_seq);
	return seq2_optimal_results;
}


void worker_process(char* seq1, int* weights, int **table)
{
	MPI_Status status;
	int len_seq1 = strlen(seq1);
	int tag, max_offset, omp_offset_end, message=1;

	int *optimal_res_omp = (int*)calloc(sizeof(int), RES_CACHE_LEN);
	int *optimal_res_CUDA = (int*)calloc(sizeof(int), RES_CACHE_LEN);
	char* seq2 = (char*)malloc(sizeof(char)*MAX_LEN_SEQ2);

	if (!optimal_res_omp || !optimal_res_CUDA|| !seq2)
	{
		printf("ERROR !optimal_res_omp || !optimal_res_CUDA|| seq2\n");
		return;
	}

	do
	{
		MPI_Recv(seq2, MAX_LEN_SEQ2, MPI_CHAR, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		tag = status.MPI_TAG;
		max_offset = len_seq1 - strlen(seq2);
		omp_offset_end = floor(max_offset/2);

		optimal_res_omp = omp_alignment_score_offset_mutant(seq1, seq2, weights, omp_offset_end, optimal_res_omp, table);
		optimal_res_CUDA = cuda_score_offset_mutant(seq1, seq2, weights, omp_offset_end, max_offset);

		if(optimal_res_omp[0] >= optimal_res_CUDA[0])
		{
			MPI_Send(&message, 1, MPI_INT, ROOT, tag, MPI_COMM_WORLD);
			MPI_Send(optimal_res_omp, 3, MPI_INT, ROOT, tag, MPI_COMM_WORLD);
		}
		else
		{
			MPI_Send(&message, 1, MPI_INT, ROOT, tag, MPI_COMM_WORLD);
			MPI_Send(optimal_res_CUDA, 3, MPI_INT, ROOT, tag, MPI_COMM_WORLD);
		}
		
	}while(tag != TERMINATE);
	//free
	free(optimal_res_CUDA);
	free(optimal_res_omp);
	free(seq2);

}


int* omp_alignment_score_offset_mutant(char* seq1,char* seq2, int* weights, int max_offset, int* optimal_res, int** table)
{
	int max_threads, thread_id;
	int index1=0, index2=0, seq2_len = strlen(seq2);
	int score=0, max_mutant = seq2_len;
	max_threads = omp_get_max_threads();

	int thread_res[max_threads][RES_CACHE_LEN];
	for(int i=0 ; i<max_threads; i++)
		thread_res[i][SCORE] = INT_MIN;
	
	optimal_res[SCORE] = INT_MIN;
		
#pragma omp parallel for shared(seq1, seq2, weights, seq2_len, max_mutant, thread_res) firstprivate(index1, index2, score) 
		for(int offset=0 ; offset < max_offset ; offset++)
		{
			thread_id = omp_get_thread_num();
			for(int mutant=1 ; mutant <= max_mutant ; mutant++)
			{
				index1=offset;
				for(index2=0 ; index2 < seq2_len ; index2++)
				{
					if(index2 == mutant && mutant != 0)
						index1++;

					score += table[seq1[index1]-'A'][seq2[index2]-'A'];

					index1++;
				}

				if(score > thread_res[thread_id][SCORE])
				{
					thread_res[thread_id][SCORE] = score;
					thread_res[thread_id][OFFSET] = offset;
					thread_res[thread_id][MUTANT] = mutant;
				}
				score=0;
			}
		}

	for(thread_id=0 ; thread_id<max_threads ; thread_id++)
	{
		if(thread_res[thread_id][SCORE]!=INT_MIN && thread_res[thread_id][SCORE] > optimal_res[SCORE])
		{
			optimal_res[SCORE] = thread_res[thread_id][SCORE];
			optimal_res[OFFSET] = thread_res[thread_id][OFFSET];
			optimal_res[MUTANT] = thread_res[thread_id][MUTANT];
		}
	}
	return optimal_res;
}


int** fill_table(int* weights, int** table)
{
	for(int i = 0 ; i < ALPHABET ; i++)
	{
		for(int j=0 ; j < ALPHABET ; j++)
		{
			if(i==j)
				table[i][j] = weights[DOLLAR];
			else if(check_group_similarity_hash(hashA,i+'A',j+'A'))
				table[i][j] = -1*weights[PRECENT];
			else if(check_group_similarity_hash(hashB,i+'A',j+'A'))
				table[i][j] = -1*weights[HASH];
			else
				table[i][j]= -1*weights[SPACE];
		}
	}
	return table;
}


int check_group_similarity_hash(char hashX[ALPHABET][ITEM_MAX_LEN], char seq1_char, char seq2_char)
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


int preprocess_and_check_input(char* secondary_sequence, int seq2_index, int seq1_len)
{
	int seq2_len = strlen(secondary_sequence);

	if(seq2_len > seq1_len || seq2_len > MAX_LEN_SEQ2)
	{
		printf("secondary_sequence[%d](%d) is larger the seq1 (%d)\n",seq2_index, seq2_len, seq1_len);
		return 0;
	}
	for(int i=0 ; i < seq2_len ; i++)
	{
		secondary_sequence[i] = toupper(secondary_sequence[i]);
		//option to check for a valid ascii a-z letter
	}
	return 1;
	
}


void print_all_optimal_results(char* seq1, int** seq2_optimal_results, int seq2_arr_size)
{
    printf("\n");
    // printf("Seq1:\n%s\n", seq1);
    printf("\nALL ALIGNMENTS:\n");
    for (int i = 0; i < seq2_arr_size; i++)
    {
		printf("SEQ2 no.%d\n",i+1);
        printf("score = %d , offset (n) = %d , mutation (k) = %d  \n", seq2_optimal_results[i][SCORE], seq2_optimal_results[i][OFFSET], seq2_optimal_results[i][MUTANT]);
        printf("\n");
    }
}


void print_table(int** table)
{
	for(int i =0;i<ALPHABET;i++)
	{
		for(int j =0;j<ALPHABET;j++)
		{
			printf("%d  ",table[i][j]);
		}
		printf("\n");
	}
}


int** mem_allocation_2d(int** pointer, int n, int m)
{
	pointer = (int**)malloc(sizeof(int*)*n);
	if(!pointer)
	{
		printf("ERROR in mem_allocation\n");
		return 0;
	}
	if (m!= 0)
	{
		for(int i=0 ; i<n ; i++)
		{
			pointer[i] = (int*)calloc(sizeof(int), m);
			if(!pointer[i])
			{
				printf("ERROR in mem_allocation[%d]\n",i);
				return 0;
			}
		}
	}
	return pointer;
}	


int conti_mem_allocation_2d(int*** array, int n, int m)
{
	/*
	allocate contiguous memory for Bcast
	*/
	int *p = (int *)malloc(n*m*sizeof(int));
    if (!p) return -1;

    /* allocate the row pointers into the memory */
    (*array) = (int **)malloc(n*sizeof(int*));
    if (!(*array)) {
       free(p);
       return -1;
    }

    /* set up the pointers into the contiguous memory */
    for (int i=0; i<n; i++) 
       (*array)[i] = &(p[i*m]);

    return 0;
}


void conti_mem_free_2d(int ***array)
{
    /* free the memory - the first element of the array is at the start */
    free(&((*array)[0][0]));

    /* free the pointers into the memory */
    free(*array);
}


// ****Sequential Main****

// int main(int argc, char *argv[])
// {
// 	double timer;
//     int my_rank, num_processes;
// 	int seq2_arr_size, seq1_len;
// 	int* weights;
// 	int** seq2_optimal_results;
//  	int** table;

//     char *seq1, **seq2_arr;

// 	seq1 = (char*)malloc(sizeof(char)*MAX_LEN_SEQ1);
// 	weights = (int*)malloc(sizeof(int)*WEIGHTS);
// 	conti_mem_allocation_2d(&table,ALPHABET, ALPHABET);

//     //MPI INIT
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &num_processes); 
	

// 	// READ INPUT
//     FILE* input = fopen(FILE_NAME, "r");
//     if(!input)
//         return -1;
    
//     // READ WEIGHTS
//     fscanf(input, "%d %d %d %d", &weights[DOLLAR], &weights[PRECENT], &weights[HASH], &weights[SPACE]);

//     // READ SEQ1
//     fscanf(input, "%s", seq1);

//     // READ THE AMOUNT OF SEQ2
//     fscanf(input, "%d", &seq2_arr_size);
    
//     // MEM ALLOC + READ ALL SEQ2
//     seq2_arr = (char**)malloc(seq2_arr_size*sizeof(char*));
//     if(!seq2_arr)
//         return -1;
    
//     for(int i=0 ; i<seq2_arr_size ; i++)
//     {
//         seq2_arr[i] = (char*)malloc(MAX_LEN_SEQ2*(sizeof(char)));
//         if(!seq2_arr[i])
//             return -1;

//         fscanf(input, "%s", seq2_arr[i]);
//     }
//     fclose(input); 


// // CHECK INPUT
//     // CHECK VALID SEQ1
//     seq1_len = strlen(seq1);
//     if(seq1_len > MAX_LEN_SEQ1)
//         return -1;

//     //CHECK IF AMOUNT OF SEQ2 IS LOWER THAN NUM PROCESSES
//     if (seq2_arr_size < num_processes - 1)
//     {
//         printf("ERROR: THE PROGRAM HAS LESS SEQ2 THAN ACTIVE PROCESSES\n");
//         return -1;
//     }

//     // UPPER CASE AND CHECK SEQ2
//     for(int i=0 ; i<seq2_arr_size ; i++)
//     {
//         if (!preprocess_and_check_input(seq2_arr[i], i, seq1_len))
//         {	
//             printf("EORROR in preprocess_and_check_input number %d\n",i);
//             return -1;
//         }
//     }

//     // MEM ALLOC FOR OPTIMAL RESULTS
//     table = fill_table(weights, table);
//     seq2_optimal_results = mem_allocation_2d(seq2_optimal_results, seq2_arr_size, RES_CACHE_LEN);

// 	int index1=0, index2=0, seq2_len;
// 	int score=0, max_mutant = seq2_len;
//     int max_offset;

    
// 	timer = MPI_Wtime();
//     for(int i=0 ; i<seq2_arr_size ; i++)
//     {
// 		seq2_optimal_results[i][SCORE] = INT_MIN;
//         seq2_len = strlen(seq2_arr[i]);
//         max_mutant = seq2_len;
//         max_offset = seq1_len - seq2_len;
//         for(int offset=0 ; offset < max_offset ; offset++)
//         {
//             for(int mutant=1 ; mutant <= max_mutant ; mutant++)
//             {
//                 index1=offset;
//                 for(index2=0 ; index2 < seq2_len ; index2++)
//                 {
//                     if(index2 == mutant && mutant != 0)
//                         index1++;

//                     score += table[seq1[index1]-'A'][seq2_arr[i][index2]-'A'];

//                     index1++;
//                 }

//                 if(score > seq2_optimal_results[i][SCORE])
//                 {
//                     seq2_optimal_results[i][SCORE] = score;
//                     seq2_optimal_results[i][OFFSET] = offset;
//                     seq2_optimal_results[i][MUTANT] = mutant;
//                 }
//                 score=0;
//             }
//         }
//     }

// 	timer = MPI_Wtime() - timer;
//     printf("\n");
// 	printf("Time to calc sequential is %lf seconds\n", timer);

//     // printf("Seq1:\n%s\n", seq1);
//     printf("\nALL ALIGNMENTS:\n");
//     for (int i = 0; i < seq2_arr_size; i++)
//     {
//         printf("SEQ2 no.%d\n",i+1);
//         printf("score = %d , offset (n) = %d , mutation (k) = %d , \n", seq2_optimal_results[i][SCORE], seq2_optimal_results[i][OFFSET], seq2_optimal_results[i][MUTANT]);
//         printf("\n");
//     }

// 	MPI_Finalize();
// 	return 0;

// }