#include <stdio.h>

int main()
{
    /* Local variables */
    double base, emin, prec, emax, rmin, rmax, t, sfmin;
    extern double dmach_dist(char *);
    double rnd, eps;

    eps = dmach_dist("Epsilon");
    sfmin = dmach_dist("Safe minimum");
    base = dmach_dist("Base");
    prec = dmach_dist("Precision");
    t = dmach_dist("Number of digits in mantissa");
    rnd = dmach_dist("Rounding mode");
    emin = dmach_dist("Minnimum exponent");
    rmin = dmach_dist("Underflow threshold");
    emax = dmach_dist("Largest exponent");
    rmax = dmach_dist("Overflow threshold");

    printf(" Epsilon                      = %e\n", eps);
    printf(" Safe minimum                 = %e\n", sfmin);
    printf(" Base                         = %.0f\n", base);
    printf(" Precision                    = %e\n", prec);
    printf(" Number of digits in mantissa = %.0f\n", t);
    printf(" Rounding mode                = %.0f\n", rnd);
    printf(" Minimum exponent             = %.0f\n", emin);
    printf(" Underflow threshold          = %e\n", rmin);
    printf(" Largest exponent             = %.0f\n", emax);
    printf(" Overflow threshold           = %e\n", rmax);
    printf(" Reciprocal of safe minimum   = %e\n", 1./sfmin);

    return 0;
}
