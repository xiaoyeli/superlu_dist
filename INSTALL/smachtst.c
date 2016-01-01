#include <stdio.h>

int main()
{
    /* Local variables */
    float base, emin, prec, emax, rmin, rmax, t, sfmin;
    extern float smach_dist(char *);
    float rnd, eps;

    eps = smach_dist("Epsilon");
    sfmin = smach_dist("Safe minimum");
    base = smach_dist("Base");
    prec = smach_dist("Precision");
    t = smach_dist("Number of digits in mantissa");
    rnd = smach_dist("Rounding mode");
    emin = smach_dist("Minnimum exponent");
    rmin = smach_dist("Underflow threshold");
    emax = smach_dist("Largest exponent");
    rmax = smach_dist("Overflow threshold");

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
