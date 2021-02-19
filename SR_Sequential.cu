#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

std::vector<int> stable_roommate(std::vector<std::vector<int>> preference_lists, int N) {
	bool stable = false;
	std::vector<int> proposal_to(N,N);
	std::vector<int> proposal_from(N,N);
	std::vector<int> proposed_to(N,0);
	std::vector<int> matching(N,0);
	
	counter = 0;

	while (!stable)
	{
		stable = true;
		for (int i = 0; i < N; i++)
		{
			// havent proposed to anyone yet
			if (proposed_to[i] >= (N-1))
			{
				return matching;
			}
			
			if (proposal_to[i] == N)
			{
				int proposee = preference_lists[i][proposed_to[i]];
				
				int op = N;
				for (int j = 0; j < preference_lists[proposee].size(); j++)
				{
					if (preference_lists[proposee][j] == i)
					{
						op = j;
						break;
					}
				}
				
				if (op == N)
				{
					return matching;
				}
				
				int op_curr = N;
				for (int j = 0; j < preference_lists[proposee].size(); j++)
				{
					if (preference_lists[proposee][j] == proposal_from[proposee])
					{
						op_curr = j;
						break;
					}
				}
				
				if (op < op_curr)
				{
					proposal_to[i] = proposee;
					
					if (proposal_from[proposee] != N)
					{
						proposal_to[proposal_from[proposee]] = N;
						stable = false;
					}
					
					proposal_from[proposee] = i;
				}
				else
				{
					stable = false;
				}
				proposed_to[i]++;
			}
		}

		counter++;
	}
	
	// remove anything that needs to be removed from phase 1
	for (int i = 0; i < N; i++)
	{
		for (int j = preference_lists[i].size()-1; j>=0; j--)
		{
			if (preference_lists[i][j] == proposal_from[i])
			{
				break;
			}
			else
			{
				if (preference_lists[i].size() == 0)
				{
					return matching;
				}
				bool erased = false;
				for (int k = 0; k < preference_lists[preference_lists[i].back()].size(); k++)
				{
					if (preference_lists[preference_lists[i].back()][k] == i)
					{
						preference_lists[preference_lists[i].back()].erase(preference_lists[preference_lists[i].back()].begin() + k);
						erased = true;
						break;
					}
				}
				if (!erased)
				{
					return matching;
				}
				preference_lists[i].pop_back();
			}
		}
	}
	
	// phase two
	stable = false;
	while (!stable)
	{
		stable = true;
		for (int i = 0; i < N; i++)
		{
			// if any list is greater than 1, need to remove rotations
			if (preference_lists[i].size() > 1)
			{
				stable = false;
				std::vector<int> x;
				std::vector<int> index;
				
				int new_index = i;
				
				int rotations_end = -1;
				
				while (rotations_end == (int)(index.end() - index.begin() - 1))
				{
					int new_x = preference_lists[new_index][1];
					new_index = preference_lists[new_x].back();
					
					rotations_end = find(index.begin(), index.end(),  new_index) - index.begin();
					
					x.push_back(new_x);
					index.push_back(new_index);
				}

				// delete rotation
				for (int j = rotations_end + 1; j < index.size(); j++)
				{
					while (preference_lists[x[j]].back() != index[j-1])
					{
						int mat_size = preference_lists[preference_lists[x[j]].back()].size();
						
						preference_lists[preference_lists[x[j]].back()].erase(std::remove(preference_lists[preference_lists[x[j]].back()].begin(),
																			preference_lists[preference_lists[x[j]].back()].end(), x[j]), 
																			preference_lists[preference_lists[x[j]].back()].end());
						
						// if not removed, no stable matching
						if (mat_size == preference_lists[preference_lists[x[j]].back()].size())
						{
							return matching;
						}
						
						// if only 1 element remaining, no stable matching
						if (preference_lists[x[j]].size() == 1)
						{
							return matching;
						}
						
						// remove
						preference_lists[x[j]].pop_back();
					}
				}
			}
		}
	}
	
	// if any list is empty
	for (int i = 0; i < preference_lists.size(); i++)
	{
		if (preference_lists[i].empty())
		{
			return matching;
		}
	}
	
	// otherwise, there are matchings
	for(int i = 0; i < N; i++)
	{
		matching[i] = preference_lists[i][0];
	}
	
	return matching;
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
	while(std::getline(f, line))
	{
	    // inner vector
		std::vector<int> row;
		std::stringstream ss(line);
		std::string data;
		// numbers are separated by commas
		while(std::getline(ss, data, ','))
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
	file.open("outp.txt", std::ios::out);
		
	// if all 0s, no matches. fill with zeros
	if (std::adjacent_find( matching.begin(), matching.end(), std::not_equal_to<>() ) == matching.end())
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