#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_randist.h> // the functions for random variates and probability density functions
#include <gsl/gsl_cdf.h> //the corresponding cumulative distribution functions
//g++ example.cpp -o example -lgsl -lgslcblas

double f (double r, void * params) {
  double alpha = *(double *) params; 
  //the expression (double *) params means "treat params as a pointer to double instead of a pointer to void"
  //the (double *) is a cast
  double f = log(alpha*r) / sqrt(r);
  return f;
}

double gauss_density (double epsilon, void * params_ptr) {
    double sigma = (*(double(*)[2]) params_ptr)[1];
    // (double(*)[2]) is casting `params_ptr` to type `an array of 2 doubles`
    // *(double(*)[2]) params_ptr is dereferencing that pointer (so the array of 2 doubles is returned)
    // (*(double(*)[2]) params_ptr)[i] so we can index the ith element of the array

    double integrand = gsl_ran_gaussian_pdf(epsilon, sigma);
    printf ("sigma is: %f\n", sigma);
    return integrand;
}

int
main (void) {
  gsl_integration_workspace * w = gsl_integration_workspace_alloc (1000);

  double result, error;
  double expected = 0.5;
  // double sigma = 1.0;
  double params[2] = {1.0, 2.0}; 
  double (*params_ptr)[2] = &params; // params_ptr is a pointer that can point to an array fo 2 doubles
                            // the base type of params_ptr is 'an array of 2 doubles'
  // params_ptr = &params; //points to the whole array params

  gsl_function F;
  F.function = &gauss_density;
  F.params = params_ptr;

  gsl_integration_qagiu(&F, 0, 1e-14, 1e-14, 1000,
                          w, &result, &error); //&F is a pointer to a gsl_function
    
  printf ("result          = % .18f\n", result);
  printf ("exact result    = % .18f\n", expected);
  printf ("estimated error = % .18f\n", error);
  printf ("actual error    = % .18f\n", result - expected);
  printf ("intervals       = %zu\n", w->size);

  gsl_integration_workspace_free (w);


//   double Q;
//   double x = 1.96*2.0;
//   double sigma = 2.0;
//   Q = gsl_cdf_gaussian_Q (x, sigma); // this function compute the cumulative distribution Q(x) = P(X > x)
//   printf ("prob(x > %f) = %f\n", x, Q);
    
//     double D;
//   double y = 0.0;
//   D = gsl_ran_gaussian_pdf(y, 1.0); //computes the probability density p(x) at y
//   printf ("density(x = %f) = %f\n", y, D);

  return 0;
}