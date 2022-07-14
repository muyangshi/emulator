module load gcc
module load gsl
module load openmpi
module load boost/1.61.0

# original
# g++ -std=c++11 -Wall -pedantic p_inte.cpp  -shared -fPIC -o p_inte.so -lgsl -lgslcblas

# Worked 1
$CXX -std=c++11 -Wall -pedantic -L$CURC_GSL_LIB -lgsl -lgslcblas -I$CURC_GSL_INC -I$CURC_BOOST_INC p_inte.cpp -shared -fPIC -o p_inte.so
 

# Worked 2
$CXX -std=c++11 -Wall -pedantic -I$CURC_GSL_INC -I$CURC_BOOST_INC p_inte.cpp -shared -fPIC -o p_inte.so -L$CURC_GSL_LIB -lgsl -lgslcblas