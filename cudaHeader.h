#pragma once

#define THREADS_PER_BLOCK 500

#define ALPHABET 26
#define ITEM_MAX_LEN 11

#define RES_CACHE_LEN 3 //The Result variable composed out of 3 values: score, offset, mutant.

#define WEIGHTS 4
#define DOLLAR 0 // Two chars are Globaly Similar (equal)
#define PRECENT 1 // Two chars similar under GroupA
#define HASH 2 // Two chars similar under GroupB
#define SPACE 3 // Two chars NOT similar

#define SCORE 0
#define OFFSET 1
#define MUTANT 2

int* cuda_score_offset_mutant(char* seq1, char* seq2 , int* weights, int offset_start, int offset_end);