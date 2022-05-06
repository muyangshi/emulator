/* p_integrand.cpp */
//g++ -std=c++11 -Wall -pedantic p_inte.cpp  -shared -fPIC -o p_inte.so -lgsl -lgslcblas
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <gsl/gsl_errno.h>

extern "C"
{

struct my_f_params {double xval; double phi; double gamma;};

/* Survival function for R^phi*W with unshifted Pareto */
int RW_marginal_C(double *xval, double phi, double gamma, int n_xval, double *result){
    double tmp2 = pow(gamma/2, phi)/boost::math::tgamma(0.5);
    double tmp1, tmp0, a;
    a = 0.5-phi;
    
    for(int i=0; i<n_xval; i++){
        tmp1 = gamma/(2*pow(xval[i],1/phi));
        tmp0 = tmp2/(a*xval[i]);
        result[i] = boost::math::gamma_p(0.5L,tmp1) + boost::math::tgamma((long double)(a+1),tmp1)*tmp0-pow(tmp1,a)*exp(-tmp1)*tmp0;
    }
    return 1;
}

/* Density function for R^phi*W with unshifted Pareto */
int RW_density_C(double *xval, double phi, double gamma, int n_xval, double *result){
    double tmp2 = pow(gamma/2, phi)/boost::math::tgamma(0.5);
    double tmp1, tmp0, a;
    a = 0.5-phi;
    
    for(int i=0; i<n_xval; i++){
        tmp1 = gamma/(2*pow(xval[i],1/phi));
        tmp0 = tmp2/(a*pow(xval[i],2));
        result[i] = (boost::math::tgamma((long double)(a+1),tmp1)-pow(tmp1,a)*exp(-tmp1))*tmp0;
    }
    return 1;
}


/* Distribution integrand */
double p_integrand(double x, void * p) {
    struct my_f_params *params = (struct my_f_params *)p;
    double xval   = (params->xval);
    double phi  = (params->phi);
    double gamma = (params->gamma);
    
    return pow(x,phi-1.5)*exp(-gamma /(2*x))/(xval+pow(x,phi));
}


/* Distribution function for R^phi*W */
double pmixture_C(double xval, double phi, double gamma){
    double result = 0.0;
    double error;
    double two_pi = boost::math::constants::two_pi<double>();
    double constant = sqrt(gamma/two_pi);
    
    gsl_integration_workspace * w
      = gsl_integration_workspace_alloc (1e4);
    gsl_set_error_handler_off();
    
    gsl_function F;
    F.function = &p_integrand;
    struct my_f_params params = { xval, phi, gamma };
    F.params = &params;
    
    int err = gsl_integration_qagiu (&F, 0, 1e-12, 1e-12, 1e4,
                          w, &result, &error);
    // Error handling: decrease epsabs if diverging
    if(err == GSL_EDIVERGE){
         err = gsl_integration_qagiu (&F, 0, 1e-7, 1e-7, 1000,
                              w, &result, &error);
      }
    
    // Error handling: still diverging
    if(err == GSL_EDIVERGE & xval > 1e4){
        err = RW_marginal_C(&xval, phi, gamma, 1, &result);
        result = result/constant;
      }
    gsl_integration_workspace_free (w);
    return 1-result*constant;
}

/* No gain compared to np.vectorize() */
//int pmixture_C_vec(double* xval, double phi, double gamma, int n_xval, double *result){
//    double error;
//    double two_pi = boost::math::constants::two_pi<double>();
//    double constant = sqrt(gamma/two_pi);
//    
//    gsl_integration_workspace * w
//      = gsl_integration_workspace_alloc (1e4);
//    
//    gsl_function F;
//    F.function = &p_integrand;
//    
//    for(int iter=0; iter < n_xval; iter++) {
//        struct my_f_params params = { xval[iter], phi, gamma };
//        F.params = &params;
//        
//        double tmp_result = 0.0;
//        gsl_integration_qagiu (&F, 0, 1e-12, 1e-10, 1e4,
//                              w, &tmp_result, &error);
//        result[iter] = 1-tmp_result*constant;
//        
//    }
//    
//    gsl_integration_workspace_free (w);
//    
//    return 1;
//}

/* Density integrand */
double d_integrand(double x, void * p) {
    struct my_f_params *params = (struct my_f_params *)p;
    double xval   = (params->xval);
    double phi  = (params->phi);
    double gamma = (params->gamma);
    
    return pow(x,phi-1.5)*exp(-gamma /(2*x))/pow(xval+pow(x,phi), 2);
}


/* Density function for R^phi*W */
double dmixture_C(double xval, double phi, double gamma){
    double result = 0.0;
    double error;
    double two_pi = boost::math::constants::two_pi<double>();
    double constant = sqrt(gamma/two_pi);
    
    gsl_integration_workspace * w
      = gsl_integration_workspace_alloc (1e4);
    gsl_set_error_handler_off();
    
    gsl_function F;
    F.function = &d_integrand;
    struct my_f_params params = { xval, phi, gamma };
    F.params = &params;
    
    int err = gsl_integration_qagiu (&F, 0, 1e-12, 1e-12, 1e4,
                          w, &result, &error);
    // Error handling: decrease epsabs if diverging
    if(err == GSL_EDIVERGE){
         err = gsl_integration_qagiu (&F, 0, 1e-7, 1e-7, 1000,
                              w, &result, &error);
      }
        
    // Error handling: still diverging
    if(err == GSL_EDIVERGE & xval > 1e4){
        err = RW_density_C(&xval, phi, gamma, 1, &result);
        result = result/constant;
      }
    
    gsl_integration_workspace_free (w);
    return result*constant;
}

/* Get the quantile range for certain probability levels */
int find_xrange_pRW_C(double min_p, double max_p, double min_x, double max_x, double phi, double gamma, double *x_range){
    if (min_x >= max_x){
        printf("Initial value of mix_x must be smaller than max_x.\n");
        exit(EXIT_FAILURE);
    }
    
    /* First the min */
    double p_min_x = pmixture_C(min_x, phi, gamma);
    while (p_min_x > min_p){
        min_x = min_x/2; /* R^phi*W is always positive */
        p_min_x = pmixture_C(min_x, phi, gamma);
    }
        
    x_range[0] = min_x;
    
    /* Now the max */
    double p_max_x = pmixture_C(max_x, phi, gamma);
    while (p_max_x < max_p){
        max_x = max_x*2; /* Upper will set to 20 initially */
        p_max_x = pmixture_C(max_x, phi, gamma);
    }
        
    x_range[1] = max_x;
    return 1;
}

/* Get the quantile using the bisection method */
double qRW_bisection_C(double p, double phi, double gamma, int n_x){
    double x_range[2];
    int tmp_res = 0;
    tmp_res = find_xrange_pRW_C(p, p, 1.0, 5.0, phi, gamma, x_range);
    double m = (x_range[0]+x_range[1])/2;
    int iter=0;
    double new_F = pmixture_C(m, phi, gamma);
    double diff = new_F-p;
    while (iter<n_x & abs(diff)> 1e-04){
        if (diff>0){
            x_range[1] = m;}
        else{
            x_range[0]=m;}
        m = (x_range[0]+x_range[1])/2;
        new_F = pmixture_C(m, phi, gamma);
        diff = new_F-p;
        iter += 1;
    }
    return m;
}




/* Get the quantile using Newton-Raphson method */
double qRW_newton_C(double p, double phi, double gamma, int n_x){
    double x_range[2];
    int tmp_res = 0;
    tmp_res = find_xrange_pRW_C(p, p, 1.0, 5.0, phi, gamma, x_range);
    double new_x, current_x = x_range[0];
    int iter=0;
    double error=1;
    double F_value, f_value;
    
    while (iter<n_x & error> 1e-08){
        F_value = pmixture_C(current_x , phi, gamma);
        f_value = dmixture_C(current_x , phi, gamma);
        new_x = current_x - (F_value-p)/f_value;
        error = abs(new_x-current_x);
        iter += 1;
        current_x = fmax(x_range[0], new_x);
        if(current_x == x_range[0]){current_x = qRW_bisection_C(p, phi, gamma, 100);}
    }
    
    return current_x;
}

}
