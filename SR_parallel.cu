#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>

int* readfile(const char* filename, int* size);
/*****************************************************
while !stable
	propose in parallel
	block
	accept/reject in parallel
	block
*****************************************************/

__device__ bool stable;
__device__ bool no_match;
__device__ bool gpu_reduced_size_empty;

__global__ void p1_proposal(int* preference_lists, int* proposal_to, int* proposed_to, int N) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	stable = true;

	if (row < N)
	{
		__syncthreads();
		if (proposed_to[row] >= (N-1))
		{
			no_match = true;
			return;
		}
		
		// if proposal was rejected, or havent proposed to anyone yet
		if (proposal_to[row] == N)
		{
			proposal_to[row] = preference_lists[row * N + proposed_to[row] + 1];
		}
	}
}

__global__ void p1_rejection(int* preference_lists, int* proposal_to, int* proposal_from, int* proposed_to, int N) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int proposee;

	if (row < N)
	{
		__syncthreads();

		proposee = proposal_to[row];
		
		int op = N;
		for (int i = 0; i < (N-1); i++)
		{
			if (preference_lists[proposee * N + i + 1] == row)
			{
				op = i;
				break;
			}
		}

		if (op == N)
		{
			no_match = true;
			return;
		}

		int op_curr = N;
		for (int i = 0; i < (N-1); i++)
		{
			if (preference_lists[proposee * N + i + 1] == proposal_from[proposee*N])
			{
				
				op_curr = i;
				break;
			}
		}

		if (op < op_curr)
		{
			if (proposal_from[proposee*N] != N)
			{
				proposal_to[proposal_from[proposee*N]] = N;
				stable = false;
			}
			proposal_from[proposee*N + op + 1] = row;
		}
		else if (op == op_curr)
		{
		}
		else
		{
			stable = false;
		}
		atomicAdd(&proposed_to[row], 1);

		__syncthreads();
	}
}

__global__ void p1_accept(int* proposal_from, int* proposal_to, int* C, int* rank, int N) {

	int i = threadIdx.x;
	int j = blockIdx.x;
	
	if (i < N && j < N)
	{
		C[j * N + i] = N;

		if (proposal_from[j * N + i] != N)
		{
			C[j * N + i] = rank[j * N + proposal_from[j * N + i]];
		}

		for (int d = 1; d < N; d *= 2)
		{
			if (i - d >= 0)
			{
				if (C[j * N + i] > C[j * N + i - d])
				{
					C[j * N + i] = C[j * N + i - d];
				}
			}
			__syncthreads();
		}

		if (C[j * N + i] != N)
		{
			proposal_from[j * N] = proposal_from[j * N + i];
		}
		if (i != 0)
		{
			proposal_from[j * N + i] = N;
		}
	}
}

__global__ void p1_evaluate(int* proposal_from, int* proposal_to, int N)
{
	int i = threadIdx.x;

	if (i < N)
	{
		proposal_to[i] = N;

		__syncthreads();

		if (proposal_from[i * N] != N) {
			proposal_to[proposal_from[i * N]] = i;
		}
		else {
			stable = false;
		}

		__syncthreads();
	}
}

__global__ void p1_remove(int* preference_lists, int* proposal_from, int* rank, int N)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if (i < (N - 1) && j < N)
	{
		if (rank[j * N + i] > rank[j * N + proposal_from[j * N]] && rank[j * N + i] != N)
		{
			preference_lists[j * N + rank[j * N + i]] = N;
			preference_lists[i * N + rank[i * N + j]] = N;
		}
	}
}

__global__ void p1_shift(int* preference_lists, int* reduced_size, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		int count = 0;

		for (int d = 0; d < N; d++)
		{
			if (preference_lists[i * N + d] != N)
			{
				preference_lists[i * N + count++] = preference_lists[i * N + d];
			}
		}

		reduced_size[i] = count - 1;
		if (reduced_size[i] <= 0)
		{
			no_match = true;
			return;
		}

		while (count < N)
		{
			preference_lists[i * N + count++] = N;
		}
	}
}

__global__ void get_rank(int* preference_lists, int N, int* rank, int* reduced_size)
 {
	int i = threadIdx.x;
	int j = blockIdx.x;

	if (i < N && j < N)
	{
		rank[j * N + i] = N;
		if (i < reduced_size[j] + 1) {
			rank[j * N + preference_lists[j * N + i]] = i;
		}
		if (i == j)
		{
			rank[j * N + i] = N;
		}
	}
	__syncthreads();
		
}
__global__ void p2_remove(int* preference_lists, int* reduced_size, int* rotations, int N, int* rank, int count)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	__syncthreads();
	if (i < count)
	{
		if (rotations[i] != N)
		{
			if (i % 2 == 1)//odd
			{
				preference_lists[rotations[i] * N + rank[rotations[i] * N + rotations[i - 1]]] = N;
				atomicSub(&reduced_size[rotations[i]], 1);

				if (reduced_size[rotations[i]] == 0)
				{
					no_match = true;
					return;
				}
			}
			else //even
			{
				preference_lists[rotations[i] * N + rank[rotations[i] * N + rotations[i + 1]]] = N;
				atomicSub(&reduced_size[rotations[i]], 1);

				if (reduced_size[rotations[i]] == 0)
				{
					no_match = true;
					return;
				}
			}
		}
	}
}


__global__ void p2_getSecondLastChoice(int* preference_lists, int* last_choice, int* second_choice, int* reduced_size, int N)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	if (i < N && j < N)
	{
		if (preference_lists[j * N + i] < N && reduced_size[j] > 1)
		{
			// second choice
			if (i == 2)
			{
				second_choice[j] = preference_lists[j * N + i];
			}

			// last choice needs to be size
			if (i == reduced_size[j])
			{
				last_choice[j] = preference_lists[j * N + i];
			}
		}
		else if (reduced_size[j] == 0)
		{
			no_match = true;
			return;
		}
	}

}

__global__ void get_reduced_sizes(int* reduced_size, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		if (reduced_size[i] <= 0)
		{
			gpu_reduced_size_empty = true;
		}
	}
}

__global__ void fill_matching_zeros(int* matching, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		matching[i] = 0;
	}
}

__global__ void fill_matching(int* preference_lists, int* matching, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		matching[i] = preference_lists[i * N + 1];
	}
}

std::vector<int> stable_roommate(std::vector<std::vector<int>> preference_lists_vector, int N) {

	int NUM_BLOCKS;
	int NUM_THREADS;

	if (N % 32)
	{
		NUM_BLOCKS = N + (32 - N % 32);
		NUM_THREADS = N + (32 - N % 32);
	}
	else
	{
		NUM_BLOCKS = N;
		NUM_THREADS = N;
	}

	int* gpu_preference_lists;
	int* gpu_proposal_to;
	int* gpu_proposal_from;
	int* gpu_proposed_to;
	int* gpu_matching;
	int* gpu_reduced_size;
	int* gpu_second_choice;
	int* gpu_last_choice;
	int* gpu_rotations;
	int* gpu_C;

	int* preference_lists = new int[N*N];

	for (int i = 0; i < N; i++)
	{
		preference_lists[i*N] = i;
		for (int j = 1; j < N; j++)
		{
			preference_lists[i*N+j] = preference_lists_vector[i][j-1];
		}
	}

	int* proposal_from = new int[N*N];
	proposal_from = (int *)calloc(N*N, sizeof(*proposal_from));
	std::replace(proposal_from, proposal_from + N*N, 0, N);

	int *proposed_to = new int[N];
	proposed_to = (int *)calloc(N, sizeof(*proposed_to));
	int *proposal_to = new int[N];
	proposal_to = (int *)calloc(N, sizeof(*proposal_to));
	std::replace(proposal_to, proposal_to + N, 0, N);
	int* reduced_size = new int[N];
	reduced_size = (int*)calloc(N, sizeof(*reduced_size));
	std::replace(reduced_size, reduced_size + N, 0, N-1);
	int* second_choice = new int[N];
	second_choice = (int*)calloc(N, sizeof(*second_choice));
	std::replace(second_choice, second_choice + N, 0, N);
	int* last_choice = new int[N];
	last_choice = (int*)calloc(N, sizeof(*last_choice));
	std::replace(last_choice, last_choice + N, 0, N);
	int* rotations = new int[N * N];
	rotations = (int*)calloc(N * N, sizeof(*rotations));
	std::replace(rotations, rotations + N * N, 0, N);
	int* C = new int[N * N];
	C = (int*)calloc(N * N, sizeof(*C));
	std::replace(C, C + N * N, 0, N);

	int *matching = new int[N];
	matching = (int *)calloc(N, sizeof(*matching));

	cudaMalloc(&gpu_preference_lists, N * N * sizeof(int));
	cudaMalloc(&gpu_proposal_to, N * sizeof(int));
	cudaMalloc(&gpu_proposal_from, N * N * sizeof(int));
	cudaMalloc(&gpu_proposed_to, N * sizeof(int));
	cudaMalloc(&gpu_matching, N * sizeof(int));
	cudaMalloc(&gpu_reduced_size, N * sizeof(int));
	cudaMalloc(&gpu_second_choice, N * sizeof(int));
	cudaMalloc(&gpu_last_choice, N * sizeof(int));
	cudaMalloc(&gpu_rotations, N * N * sizeof(int)); 
	cudaMalloc(&gpu_C, N * N * sizeof(int));
	
	bool stable_host;
	bool no_match_host;
	bool reduced_size_empty;

	stable_host = false;
	no_match_host = false;
	cudaMemcpy(gpu_preference_lists, preference_lists, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_proposal_to, proposal_to, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_proposal_from, proposal_from, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_proposed_to, proposed_to, N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_C, C, N * N * sizeof(int), cudaMemcpyHostToDevice);

	int* rank;
	cudaMalloc(&rank, N * N * sizeof(int));
	cudaMemcpy(gpu_reduced_size, reduced_size, N * sizeof(int), cudaMemcpyHostToDevice);
	get_rank <<<NUM_BLOCKS, NUM_THREADS >>> (gpu_preference_lists, N, rank, gpu_reduced_size);

	while (!(stable_host) && !(no_match_host))
	{
		p1_proposal<<<1, NUM_THREADS>>> (gpu_preference_lists, gpu_proposal_to, gpu_proposed_to, N);
		cudaDeviceSynchronize();

		p1_rejection<<<1, NUM_THREADS>>> (gpu_preference_lists, gpu_proposal_to, gpu_proposal_from, gpu_proposed_to, N);
		cudaDeviceSynchronize();

		p1_accept<<<NUM_BLOCKS, NUM_THREADS>>> (gpu_proposal_from, gpu_proposal_to, gpu_C, rank, N);
		cudaDeviceSynchronize();

		p1_evaluate <<<1, NUM_THREADS>>> (gpu_proposal_from, gpu_proposal_to, N);

		cudaMemcpyFromSymbol(&stable_host, stable, sizeof(stable_host), 0, cudaMemcpyDeviceToHost);
		cudaMemcpyFromSymbol(&no_match_host, no_match, sizeof(no_match_host), 0, cudaMemcpyDeviceToHost);
	}

	if (!no_match_host)
	{
		int* rank;
		cudaMalloc(&rank, N * N * sizeof(int));
		cudaMemcpy(gpu_reduced_size, reduced_size, N * sizeof(int), cudaMemcpyHostToDevice);
		get_rank <<<NUM_BLOCKS, NUM_THREADS>>> (gpu_preference_lists, N, rank, gpu_reduced_size);
		p1_remove <<<NUM_BLOCKS, NUM_THREADS>>> (gpu_preference_lists, gpu_proposal_from, rank, N);
		cudaDeviceSynchronize();
		cudaMemcpy(gpu_reduced_size, reduced_size, N * sizeof(int), cudaMemcpyHostToDevice);
		p1_shift <<<1, NUM_THREADS>>> (gpu_preference_lists, gpu_reduced_size, N);
		cudaDeviceSynchronize();
		cudaMemcpyFromSymbol(&no_match_host, no_match, sizeof(no_match_host), 0, cudaMemcpyDeviceToHost);
		p2_getSecondLastChoice <<<NUM_BLOCKS, NUM_THREADS>>> (gpu_preference_lists, gpu_last_choice, gpu_second_choice, gpu_reduced_size, N);
		cudaDeviceSynchronize();
	}

	stable_host = false;

	// get rotations
	while (!(stable_host) && !(no_match_host))
	{
		stable_host = true;

		for (int i = 0; i < N; i++)
		{
			cudaMemcpy(reduced_size, gpu_reduced_size, N * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(second_choice, gpu_second_choice, N * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(last_choice, gpu_last_choice, N * sizeof(int), cudaMemcpyDeviceToHost);
			rotations = (int*)calloc(N * N, sizeof(*rotations));
			std::replace(rotations, rotations + N * N, 0, N);

			if (reduced_size[i] > 1)
			{
				stable_host = false;
				int counter = 0;
				bool cycle_complete = false;
				rotations[counter] = second_choice[i];
				counter++;
				while (!cycle_complete)
				{
					if (counter % 2 == 0)
					{
						rotations[counter] = second_choice[rotations[counter-1]];
					}
					else
					{
						rotations[counter] = last_choice[rotations[counter-1]];
					}

					if (rotations[counter] == i)
					{
						cycle_complete = true;
					}
					counter++;
				}
				
				int* rank;
				int num_threads_counter;

				if (counter % 32)
				{
					num_threads_counter = counter + (32 - counter % 32);
				}
				else
				{
					num_threads_counter = counter;
				}

				cudaMalloc(&rank, N * N * sizeof(int));
				cudaMemcpy(gpu_rotations, rotations, N * N * sizeof(int), cudaMemcpyHostToDevice);
				get_rank <<<NUM_BLOCKS, NUM_THREADS>>> (gpu_preference_lists, N, rank, gpu_reduced_size);
				cudaDeviceSynchronize();
				p2_remove <<<1, num_threads_counter>>> (gpu_preference_lists, gpu_reduced_size, gpu_rotations, N, rank, counter);
				cudaDeviceSynchronize();
				p1_shift <<<1, NUM_THREADS>>> (gpu_preference_lists, gpu_reduced_size, N);
				cudaDeviceSynchronize();
				p2_getSecondLastChoice <<<NUM_BLOCKS, NUM_THREADS>>> (gpu_preference_lists, gpu_last_choice, gpu_second_choice, gpu_reduced_size, N);
				cudaDeviceSynchronize();
				cudaMemcpyFromSymbol(&no_match_host, no_match, sizeof(no_match_host), 0, cudaMemcpyDeviceToHost);
			}
		}
	}

	reduced_size_empty = false;

	get_reduced_sizes <<<1, NUM_THREADS>>> (gpu_reduced_size, N);
	cudaMemcpyFromSymbol(&reduced_size_empty, gpu_reduced_size_empty, sizeof(reduced_size_empty), 0, cudaMemcpyDeviceToHost);

	cudaMemcpy(gpu_matching, matching, N * sizeof(int), cudaMemcpyHostToDevice);

	if (no_match_host || reduced_size_empty)
	{
		fill_matching_zeros <<<1, NUM_THREADS>>> (gpu_matching, N);
	}
	else
	{
		fill_matching <<<1, NUM_THREADS>>> (gpu_preference_lists, gpu_matching, N);
	}
	
	cudaDeviceSynchronize();
	cudaMemcpy(matching, gpu_matching, N * sizeof(int), cudaMemcpyDeviceToHost);

	std::vector<int> matching_vector(matching, matching + N);

	cudaFree(gpu_preference_lists);
	cudaFree(gpu_proposal_to);
	cudaFree(gpu_proposal_from);
	cudaFree(gpu_proposed_to);
	cudaFree(gpu_matching);
	cudaFree(gpu_reduced_size);
	cudaFree(gpu_second_choice);
	cudaFree(gpu_last_choice);
	cudaFree(gpu_rotations);

	return matching_vector;
}

int main()
{
	// 2d vector for the preference lists
	std::vector<std::vector<int>> preference_lists;
	int N = 0;
	std::vector<int> matching;

	// input file
	std::ifstream f("inp.txt");
	// get line
	std::string line;

	// while another line to get
	while (std::getline(f, line))
	{
		// inner vector
		std::vector<int> row;
		std::stringstream ss(line);
		std::string data;
		// numbers are separated by commas
		while (std::getline(ss, data, ','))
		{
			// put numbers in vector
			row.push_back(std::stoi(data));
		}
		// put vector in 2d vector
		preference_lists.push_back(row);
		N++;
	}

	matching = stable_roommate(preference_lists, N);

	// output to file
	std::fstream file;
	file.open("outp_p.txt", std::ios::out);

	// if all 0s, no matches. fill with zeros
	if (std::adjacent_find(matching.begin(), matching.end(), std::not_equal_to<>()) == matching.end())
	{
		// print results to text file
		file << "NULL" << "\n";
	}
	else
	{
		for (int i = 0; i < matching.size(); i++)
		{
			file << matching[i] << "\n";
		}
	}
	file.close();

	return 0;
}