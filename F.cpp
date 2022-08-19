/* p_integrand.cpp */
// g++ -std=c++11 -Wall -pedantic F.cpp -o F -lgsl -lgslcblas
// ./F
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_randist.h> // the functions for random variates and probability density functions
#include <gsl/gsl_cdf.h> //the corresponding cumulative distribution functions
#include <iostream>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_math.h>
#include <limits>

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

/* ----------------------------------------------------------------------------------------- 
 * Calculate F_X(x), with the addition of the nugget term epsilon ~ N(0, \tau^2)
 * -----------------------------------------------------------------------------------------
 */

// double first_integrand (double epislon, void * params) { // inte_x^\inf \phi(\epsilon) d\epsilon
//     // double params[] = { xval, phi, gamma, tau };
//     double tau = (*(double[] *) params)[3];
//     double tau = *(double *) params;
//     double integrand = gsl_ran_gaussian_pdf(epsilon, tau);
//     return integrand
// }

double F_X_star_integrand (double r, void * params_ptr) {
    double xval  = (*(double(*)[3]) params_ptr)[0];
    double phi   = (*(double(*)[3]) params_ptr)[1];
    double gamma = (*(double(*)[3]) params_ptr)[2];
    double integrand = sqrt(gamma/(2*M_PI))*pow(r,phi-1.5)*exp(-gamma/(2*r))/(xval+pow(r,phi));
    return integrand;
}

double F_X_star (double xval, double phi, double gamma) {
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);

    double result, error;
    double params[3] = { xval, phi, gamma };

    gsl_function F;
    F.function = &F_X_star_integrand;
    F.params = &params;

    gsl_integration_qagiu(&F, 0, 1e-12, 1e-12, 10000,
                            w, &result, &error);
    gsl_integration_workspace_free (w);
    return 1-result;
}

double F_X_first_integrand (double epsilon, void * params_ptr) {
    double tau = (*(double(*)[4]) params_ptr)[3];
    double integrand = gsl_ran_gaussian_pdf(epsilon, tau);
    return integrand;
}

double F_X_second_integrand (double epsilon, void * params_ptr) { // phi(epsilon) * pmixture_C(xval-epsilon, phi, gamma)
    double xval  = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    double tau   = (*(double(*)[4]) params_ptr)[3];
    double integrand = gsl_ran_gaussian_pdf(epsilon, tau)*(1.0 - pmixture_C(xval-epsilon, phi, gamma));
    // double integrand = gsl_ran_gaussian_pdf(epsilon, tau)*(1.0 - F_X_star(xval-epsilon, phi, gamma));
    return integrand;
}

double F_X_part_1 (double xval, double phi, double gamma, double tau) {
    double result_part_1;
    result_part_1 = gsl_cdf_gaussian_Q (xval, tau);
    return result_part_1;
}

double F_X_part_2 (double xval, double phi, double gamma, double tau) {
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);

    double result, error;
    double params[4] = { xval, phi, gamma, tau }; // params is an array of 4 doubles
    // double (*params_ptr)[4] = &params; // `params_ptr` is a pointer that point to `an array of 4 doubles`,
                                        // the base type of `params_ptr` is `an array of 4 doubles`

    gsl_function F_X_part_2;
    F_X_part_2.function = &F_X_second_integrand;
    F_X_part_2.params = &params;

    // int gsl_integration_qagil(gsl_function *f, double b, double epsabs, double epsrel, size_t limit, 
    //                              gsl_integration_workspace *workspace, double *result, double *abserr)

    gsl_integration_qagil(&F_X_part_2, xval, 1e-12, 1e-12, 10000,
                            w, &result, &error);

    gsl_integration_workspace_free (w);
    return result;
}

double find_lower_bound (double tau) {
    double NEGATIVE_HUGE = -std::numeric_limits<double>::max();
    double LB = NEGATIVE_HUGE;
    double new_LB = LB/2;
    while (gsl_ran_gaussian_pdf(LB, tau) == 0) {
        if (gsl_ran_gaussian_pdf(new_LB, tau) == 0) {
            LB = new_LB;
            new_LB = new_LB/2;
        } else {
            LB = (new_LB + LB)/2;
        }
    }
    // std::cout << "LB: " << LB << "\n";
    // std::cout << "density: " << gsl_ran_gaussian_pdf(LB, tau) << "\n";
    return LB;
}

double F_X_part_2_QAG (double xval, double phi, double gamma, double tau) {
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);

    double result, error;
    double params[4] = { xval, phi, gamma, tau }; // params is an array of 4 doubles
    // double (*params_ptr)[4] = &params; // `params_ptr` is a pointer that point to `an array of 4 doubles`,
                                        // the base type of `params_ptr` is `an array of 4 doubles`

    gsl_function F_X_part_2;
    F_X_part_2.function = &F_X_second_integrand;
    F_X_part_2.params = &params;

    // int gsl_integration_qagil(gsl_function *f, double b, double epsabs, double epsrel, size_t limit, 
    //                              gsl_integration_workspace *workspace, double *result, double *abserr)

    double LB = find_lower_bound(tau); // -38 * tau

    gsl_integration_qag(&F_X_part_2, LB, xval, 1e-12, 1e-12, 10000,
                        6, w, &result, &error);

    gsl_integration_workspace_free (w);
    return result;
}

double F_X_part_2_CQUAD (double xval, double phi, double gamma, double tau) {
    gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc (10000);

    double result, error;
    double params[4] = { xval, phi, gamma, tau};
    // double POSITIVE_INFTY = std::numeric_limits<double>::infinity();
    // double NEGATIVE_INFTY = -std::numeric_limits<double>::infinity();
    size_t neval;

    gsl_function F_X_part_2;
    F_X_part_2.function = &F_X_second_integrand;
    F_X_part_2.params = &params;

    double LB = find_lower_bound(tau);

    gsl_integration_cquad(&F_X_part_2, LB, xval, 1e-12, 1e-12,
                            w, &result, &error, &neval);
    gsl_integration_cquad_workspace_free (w);
    return result;
}

double F_X_part_2_QAGS_inner_integrand (double r, void * params_ptr) { // params = { t, phi, gamma, tau }
    double t     = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    // double tau   = (*(double(*)[4]) params_ptr)[3];
    double integrand = pow(r,phi-1.5)*exp(-gamma/(2*r))/(((1-t)/t)+pow(r,phi));
    return integrand;
}

double F_X_part_2_QAGS_inner (double t, double phi, double gamma, double tau) {
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);

    double result, error;
    double params[4] = { t, phi, gamma, tau };

    gsl_function F;
    F.function = &F_X_part_2_QAGS_inner_integrand;
    F.params = &params;

    gsl_integration_qagiu(&F, 0, 1e-12, 1e-12, 10000,
                            w, &result, &error);
    gsl_integration_workspace_free (w);
    return result;
}

double F_X_part_2_QAGS_integrand (double t, void * params_ptr) {
    double x     = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    double tau   = (*(double(*)[4]) params_ptr)[3];
    double integrand = (1/pow(t,2))*gsl_ran_gaussian_pdf(x - ((1-t)/t), tau)*F_X_part_2_QAGS_inner(t, phi, gamma, tau);
    return integrand;
}

double F_X_part_2_QAGS (double xval, double phi, double gamma, double tau) {
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);

    double result, error;
    double params[4] = {xval, phi, gamma, tau};

    gsl_function F;
    F.function = &F_X_part_2_QAGS_integrand;
    F.params = &params;

    gsl_integration_qags(&F, 0, 1, 1e-12, 1e-12, 10000,
                            w, &result, &error);
    gsl_integration_workspace_free (w);
    return sqrt(gamma/(2*M_PI))*result;
}

double F_X (double xval, double phi, double gamma, double tau) {
    double result_part_1 = F_X_part_1(xval, phi, gamma, tau);
    // double result_part_2 = F_X_part_2(xval, phi, gamma, tau);
    double result_part_2 = F_X_part_2_QAG(xval, phi, gamma, tau);
    // double result_part_2 = F_X_part_2_CQUAD(xval, phi, gamma, tau);
    // double result_part_2 = F_X_part_2_QAGS(xval, phi, gamma, tau);
    double F = 1 - (result_part_1 + result_part_2);
    return F;
}

double F_X_cheat (double xval, double phi, double gamma, double tau) {
    double F = F_X (xval, phi, gamma, tau);
    if (F == 1) {
        F = pmixture_C (xval, phi, gamma);
    }
    return F;
}

/* ---------------------------------------------------- */
/* find the quantile (x value) corresponding to F_X = p */
/* ---------------------------------------------------- */


/* integrand of \int_{-\infty}^x \varphi(\epsilon) f_{X^*}(x-\epsilon)d\epsilon */
double f_X_integrand (double epsilon, void * params_ptr) { // params is an array of 5 doubles {p, phi, gamma, tau, x}
    // double p     = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[5]) params_ptr)[1];
    double gamma = (*(double(*)[5]) params_ptr)[2];
    double tau   = (*(double(*)[5]) params_ptr)[3];
    double x     = (*(double(*)[5]) params_ptr)[4];
    return gsl_ran_gaussian_pdf(epsilon, tau)*dmixture_C(x-epsilon, phi, gamma);
}

/* the function to find the root of */
double function_to_solve (double x, void * params_ptr) {
    double p     = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    double tau   = (*(double(*)[4]) params_ptr)[3];

    // return F_X(x, phi, gamma, tau) - p;
    return F_X_cheat(x, phi, gamma, tau) - p;
}

double function_to_solve_df (double x, void * params_ptr) {
    double p     = (*(double(*)[4]) params_ptr)[0];
    double phi   = (*(double(*)[4]) params_ptr)[1];
    double gamma = (*(double(*)[4]) params_ptr)[2];
    double tau   = (*(double(*)[4]) params_ptr)[3];

    double params[5] = { p, phi, gamma, tau, x};
    gsl_integration_workspace * w = gsl_integration_workspace_alloc (10000);
    double result, error;
    gsl_function f_X;
    f_X.function = &f_X_integrand;
    f_X.params = &params;
    double LB = -38 * tau;
    gsl_integration_qag(&f_X, LB, x, 1e-12, 1e-12, 10000,
                        6, w, &result, &error);
    gsl_integration_workspace_free (w);
    return result;
}

void function_to_solve_fdf (double x, void * params_ptr,
                              double * f, double * df){
    *f = function_to_solve(x,params_ptr);
    *df = function_to_solve_df(x,params_ptr);
}

double quantile_F_X (double p, double phi, double gamma, double tau) {
    // if (p > 0.99) {
    //     printf ("p is: % .10f \n", p);
    //     double qRW = qRW_newton_C(p, phi, gamma, 100);
    //     // printf ("qRW is: % .10f \n", qRW);
    //     return qRW;
    // }
    // std::cout << "i'm here" << "\n";
    int status;
    int iter = 0, max_iter = 100;
    const gsl_root_fdfsolver_type *T; // Root Finding Algorithms using Derivatives
    gsl_root_fdfsolver *s; // A workspace for finding roots using methods that require derivatives
    gsl_function_fdf FDF; // a general function with parameters and its first derivative

    double x0, x = qRW_newton_C(p, phi, gamma, 100);

    if (F_X(x, phi, gamma, tau) == 1) {
        std::cout << "F_X failed at p = " << p << "\n";
        return x;
    }

    double params[4] = { p, phi, gamma, tau }; // params is an array of 4 doubles
    FDF.f = &function_to_solve;
    FDF.df = &function_to_solve_df;
    FDF.fdf = &function_to_solve_fdf;
    FDF.params = &params;

    /* 
        int gsl_root_fdfsolver_set(gsl_root_fdfsolver *s, gsl_function_fdf *fdf, double root)
        This function initializes, or reinitializes, an existing solver s 
        to use the function and derivative fdf and the initial guess root.
    */
    T = gsl_root_fdfsolver_newton;
    s = gsl_root_fdfsolver_alloc(T);
    gsl_root_fdfsolver_set(s, &FDF, x);

    printf ("using %s method: \n",
            gsl_root_fdfsolver_name (s));


    /*
        int gsl_root_test_delta(doublex1, double x0, double epsabs, double epsrel)
        This function tests for the convergence of the sequence `x0`, `x1` with absolute error `epsabs`
        and relative error `epsrel`. The test returns `GSL_SUCCESS` if the condition:
            |x1 - x0| < epsabs + epsrel|x1|
        is achieved, and returns `GSL_CONTINUE` otherwise.
    */
    do {
        iter++;
        status = gsl_root_fdfsolver_iterate(s);
        x0 = x;
        x = gsl_root_fdfsolver_root(s);
        status = gsl_root_test_delta(x,x0,0,1e-4);

        if (status == GSL_SUCCESS)
            printf ("Converged:\n");
        printf ("%5d %10.7f %10.7f\n",
                iter, x, x-x0);
    } while (status == GSL_CONTINUE && iter < max_iter);

    gsl_root_fdfsolver_free (s);
    return x;
}


}

int main(void){
    // double xval = 38;
    // double phi = 1;
    // double gamma = 0.5;
    // double tau = 1;

    // double integral_1 = nugget_F_part_1(xval, phi, gamma, tau);
    // double integral_2 = nugget_F_part_2(xval, phi, gamma, tau);
    // double inner_integrand = 1 - pmixture_C(xval, phi, gamma);

    // std::cout << "integral_1 is: " << integral_1 << "\n";
    // std::cout << "integral_2 is: " << integral_2 << "\n";
    // std::cout << "inner_integrand is: " << inner_integrand << "\n";
    // std::cout << "gsl_ran_gaussian_pdf(36,1.0): " << gsl_ran_gaussian_pdf(36,1.0) << "\n";
    // std::cout << "gsl_ran_gaussian_pdf(37,1.0): " << gsl_ran_gaussian_pdf(37,1.0) << "\n";
    // std::cout << "gsl_ran_gaussian_pdf(38,1.0): " << gsl_ran_gaussian_pdf(38,1.0) << "\n";
    // std::cout << "gsl_ran_gaussian_pdf(39,1.0): " << gsl_ran_gaussian_pdf(39,1.0) << "\n";
    // std::cout << "gsl_ran_gaussian_pdf(40,1.0): " << gsl_ran_gaussian_pdf(40,1.0) << "\n";

    double xval = 1;
    double phi = 1;
    double gamma = 2;
    double tau = 200;

    double F_X_value = 0;

    // double LB = find_lower_bound(tau);
    // std::cout << "LB: " << LB << "\n";

    // for (int i = LB; i > 2*LB; i-=tau) {
    //     std::cout << i << ": ";
    //     std::cout << gsl_ran_gaussian_pdf(i, tau) << "\n";
    // }

    // std::cout << "38 tau: " << gsl_ran_gaussian_pdf(38*tau, tau) << "\n";
    // std::cout << "39 tau: " << gsl_ran_gaussian_pdf(39*tau, tau) << "\n";

    // while (F_X_value != 1) {
    //     F_X_value = F_X (xval, phi, gamma, tau);
    //     std::cout << std::setprecision(10) << std::fixed;
    //     std::cout << "xval: " << xval << " and ";
    //     std::cout << "F_X(xval,phi,gamma,tau) is: " << F_X_value << "\n";
    //     xval++;
    // }

    while (F_X_value != 1 && xval < 2e14) {
        // F_X_value = F_X(xval, phi, gamma, tau);
        double F_X_star_value = pmixture_C(xval, phi, gamma);
        double F_X_value = F_X_cheat (xval, phi, gamma, tau);
        std::cout << std::setprecision(10) << std::fixed;
        std::cout << "xval: " << xval << "\n";
        std::cout << "F_X  is: " << F_X_value << "\n";
        std::cout << "F_X* is: " << F_X_star_value << "\n";
        std::cout << "real F_X is: " << F_X (xval, phi, gamma, tau) << "\n";
        // std::cout << "F_Xc is: " << F_X_cheat_value << "\n";

        double quantile_F_X_value = quantile_F_X(F_X_value, phi, gamma, tau);
        double quantile_F_X_star_value = qRW_newton_C(F_X_star_value, phi, gamma, 100);
        std::cout << "quantile_F_X : " << quantile_F_X_value << "\n";
        std::cout << "quantile_F_X*: " << quantile_F_X_star_value << "\n";

        std::cout << "\n";
        xval = xval*2;
        // xval++;
    }

    // F_X_value = F_X(5842539116055501347301368646571393024.0, phi, gamma, tau);
    // std::cout << std::setprecision(40) << std::fixed;
    // std::cout << F_X_value << "\n";

    // xval = 37;
    // F_X_value = F_X(xval, phi, gamma, tau);
    // double F_X_star_value = pmixture_C(xval, phi, gamma);
    // std::cout << std::setprecision(10) << std::fixed;
    // std::cout << "xval: " << xval << "\n";
    // std::cout << "F_X  is: " << F_X_value << "\n";
    // std::cout << "F_X* is: " << F_X_star_value << "\n";
    // double quantile_F_X_value = quantile_F_X(F_X_value, phi, gamma, tau);
    // double quantile_F_X_star_value = qRW_newton_C(F_X_star_value, phi, gamma, 100);
    // std::cout << "quantile_F_X : " << quantile_F_X_value << "\n";
    // std::cout << "quantile_F_X*: " << quantile_F_X_star_value << "\n";
    // std::cout << "test: " << qRW_newton_C(F_X_value, phi, gamma, 100) << "\n";

    // double params[4] = {0.7528135606, phi, gamma, tau};
    // double df = function_to_solve_df (37, &params);
    // std::cout << df << "\n";


    // F_X_value = F_X (37, phi, gamma, tau);
    // std::cout << std::setprecision(40) << std::fixed;
    // std::cout << F_X_value;

    // double F_X_value = F_X (xval, phi, gamma, tau);
    // double F_X_star_value = pmixture_C (xval, phi, gamma);
    // std::cout << "xval: " << xval << " phi: " << phi << " gamma: " << gamma << " tau: " << tau << "\n";
    // std::cout << "F_X(xval,phi,gamma,tau) is: " << F_X_value << "\n";
    // std::cout << "F_X_star(xval,phi,gamma,tau) is: " << F_X_star_value << "\n";
    // std::cout << "quantile_F_X is: " << quantile_F_X (F_X_value,phi,gamma,tau) << "\n";
    // std::cout << "qRW_newton_C is: " << qRW_newton_C (F_X_star_value,phi,gamma,100) << "\n";

    // std::cout << gsl_ran_gaussian_pdf(-10,tau) << "\n";

    // double POSITIVE_INFTY = std::numeric_limits<double>::max();

    // std::cout << (0 + POSITIVE_INFTY)/2/2 << "\n";
    // std::cout << "POSITIVE_INFTY is: " << POSITIVE_INFTY << "\n";
    // double NEGATIVE_INFTY = -std::numeric_limits<double>::infinity();
    // std::cout << "NEGATIVE_INFTY is: " << NEGATIVE_INFTY << "\n";
    return 0;
}
