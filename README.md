# Forked from likun-stat/nonstat_Pareto1

## Compile p_inte.cpp
`g++ -std=c++11 -Wall -pedantic p_inte.cpp  -shared -fPIC -o p_inte.so -lgsl -lgslcblas`

## Files
- calculated data (npy, npz) and trained emulators are stored in `./data` directory
- Some practice of training emulators are stored in `./resources` directory

# Original README below
## nonstat_Pareto1
### Change standard Pareto to (standard Pareto-1)
