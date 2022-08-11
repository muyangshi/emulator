/* p_integrand.cpp */
//g++ -std=c++11 -Wall -pedantic p_inte.cpp  -shared -fPIC -o p_inte.so -lgsl -lgslcblas
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_randist.h> // the functions for random variates and probability density functions
#include <gsl/gsl_cdf.h> //the corresponding cumulative distribution functions

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

double p_integrand1(double x, void * p) {
    struct my_f_params *params = (struct my_f_params *)p;
    double xval   = (params->xval);
    double phi  = (params->phi);
    double gamma = (params->gamma);
    
    double tmp0 = 1/(1-x);
    double tmp1 = 1/phi;
    double tmp = pow(xval*x*tmp0, -tmp1);
    return sqrt(tmp)*exp(-gamma*tmp/2)*tmp1*tmp0;
}


/* Distribution function for R^phi*W */
double pmixture_C_inf_integration(double xval, double phi, double gamma){
    double result = 0.0;
    double error;
    double two_pi = boost::math::constants::two_pi<double>();
    double constant = sqrt(gamma/two_pi);
    
    if (xval < 1e-10){
        return xval*constant*pow(gamma/2, -phi-0.5)*boost::math::tgamma(phi+0.5);
    }
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
    if(((err == GSL_EDIVERGE) & (xval > 1e4))|(result < 0)){
        err = RW_marginal_C(&xval, phi, gamma, 1, &result);
        result = result/constant;
      }
    gsl_integration_workspace_free (w);
    return 1-result*constant;
}

double pmixture_C(double xval, double phi, double gamma){
    double result = 0.0;
    double error = 0.0;
    double two_pi = boost::math::constants::two_pi<double>();
    double constant = sqrt(gamma/two_pi);
    
    if (xval < 1e-10){
        return xval*constant*pow(gamma/2, -phi-0.5)*boost::math::tgamma(phi+0.5);
    }
    gsl_integration_workspace * w
      = gsl_integration_workspace_alloc (1e4);
    gsl_set_error_handler_off();
    struct my_f_params params = { xval, phi, gamma };
    int err = 0;
    if(xval > 10){
          gsl_function F;
          F.function = &p_integrand1;
          F.params = &params;
          
           err = gsl_integration_qags (&F, 0.0, 1.0, 1e-14, 1e-14, 1e4,
                                w, &result, &error);
          // Error handling: decrease epsabs if diverging
          if(err == GSL_EDIVERGE){
               err = gsl_integration_qags (&F, 0.0, 1.0, 1e-7, 1e-7, 1000,
                                    w, &result, &error);
            }
          // Error handling: still diverging
          if(((err == GSL_EDIVERGE) & (xval > 1e4))|(result < 0)){
              err = RW_marginal_C(&xval, phi, gamma, 1, &result);
              result = result/constant;
            }
          result = 1-result*constant;
    }else{
          result = pmixture_C_inf_integration(xval, phi, gamma);
    }
    
    
    
    gsl_integration_workspace_free (w);
    return result;
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
    double pi = boost::math::constants::pi<double>();
    double constant = sqrt(gamma/two_pi);
    
    if (xval < 1e-8){
        double tmp = 2/gamma;
        double tmp1 = pow(tmp,phi)*boost::math::tgamma(phi+0.5) + xval*xval*pow(tmp,3*phi)*boost::math::tgamma(3*phi+0.5) - 2*xval*pow(tmp,2*phi)*boost::math::tgamma(2*phi+0.5);
        return tmp1/sqrt(pi);
    }
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
    if(((err == GSL_EDIVERGE) & (xval > 1e4))|(result < 0)){
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
//    int tmp_res = 0;
    find_xrange_pRW_C(p, p, 1.0, 5.0, phi, gamma, x_range);
    double m = (x_range[0]+x_range[1])/2;
    int iter=0;
    double new_F = pmixture_C(m, phi, gamma);
    double diff = new_F-p;
    while ((iter<n_x) & (abs(diff)> 1e-12)){
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
//    int tmp_res = 0;
//    if(p < 1e-15) {return 5.32907052e-15;}
    find_xrange_pRW_C(p, p, 1.0, 5.0, phi, gamma, x_range);
    double new_x, current_x = x_range[0];
    int iter=0;
    double error=1;
    double F_value, f_value;
    
    /* Newton method constantly diverges for large quantile levels */
    if(p>0.99) {current_x = qRW_bisection_C(p, phi, gamma, 100); return current_x;}
    
    while ((iter<n_x) & (error> 1e-08)){
        F_value = pmixture_C(current_x , phi, gamma);
        f_value = dmixture_C(current_x , phi, gamma);
        new_x = current_x - (F_value-p)/f_value;
        error = abs(new_x-current_x);
        iter += 1;
        current_x = fmax(x_range[0], new_x);
        if(current_x == x_range[0]){current_x = qRW_bisection_C(p, phi, gamma, 100);break;}
    }
    return current_x;
}

//------------------------------------------------------------------------------------------
/* Calculate F_X(x) */

// double first_integrand (double epislon, void * params) { // inte_x^\inf \phi(\epsilon) d\epsilon
//     // double params[] = { xval, phi, gamma, tao };
//     double tao = (*(double[] *) params)[3];
//     double tao = *(double *) params;
//     double integrand = gsl_ran_gaussian_pdf(epsilon, tao);
//     return integrand
// }

double nugget_F_second_integrand (double epsilon, void * params_ptr) { // phi(epsilon) * pmixture_C(xval-epsilon, phi, gamma)
    double xval  = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    double tao   = (*(double(*)[4]) params_ptr)[3];
    double integrand = gsl_ran_gaussian_pdf(epsilon, tao)*(1.0 - pmixture_C(xval-epsilon, phi, gamma));
    return integrand;
}


double nugget_F (double xval, double phi, double gamma, double tao){
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);

    /* Calculate part 2 integral */

    double result_part_2, error_part_2;
    double params[4] = { xval, phi, gamma, tao }; // params is an array of 4 doubles
    // double (*params_ptr)[4] = &params; // `params_ptr` is a pointer that point to `an array of 4 doubles`,
                                        // the base type of `params_ptr` is `an array of 4 doubles`

    gsl_function nugget_F_part_2;
    nugget_F_part_2.function = &nugget_F_second_integrand;
    nugget_F_part_2.params = &params;

    // int gsl_integration_qagil(gsl_function *f, double b, double epsabs, double epsrel, size_t limit, 
    //                              gsl_integration_workspace *workspace, double *result, double *abserr)

    gsl_integration_qagil(&nugget_F_part_2, xval, 1e-14, 1e-14, 10000,
                            w, &result_part_2, &error_part_2);

    gsl_integration_workspace_free (w);

    /* Calculate part 1 integral */

    double result_part_1;
    result_part_1 = gsl_cdf_gaussian_Q (xval, tao);

    double F;
    F = 1 - (result_part_1 + result_part_2);

    return F;
}


}
