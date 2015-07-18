#include <stdio.h>
#include "f2c.h"

/* Complex Division c = a/b */
void z_div(doublecomplex *c, doublecomplex *a, doublecomplex *b)
{
    double ratio, den;
    double abr, abi, cr, ci;
  
    if( (abr = b->r) < 0.)
	abr = - abr;
    if( (abi = b->i) < 0.)
	abi = - abi;
    if( abr <= abi ) {
	if (abi == 0) {
	    fprintf(stderr, "z_div.c: division by zero");
	    exit(-1);
	}	  
	ratio = b->r / b->i ;
	den = b->i * (1 + ratio*ratio);
	cr = (a->r*ratio + a->i) / den;
	ci = (a->i*ratio - a->r) / den;
    } else {
	ratio = b->i / b->r ;
	den = b->r * (1 + ratio*ratio);
	cr = (a->r + a->i*ratio) / den;
	ci = (a->i - a->r*ratio) / den;
    }
    c->r = cr;
    c->i = ci;
}

/* Return the complex conjugate */
void d_cnjg(doublecomplex *r, doublecomplex *z)
{
    r->r = z->r;
    r->i = -z->i;
}

/* Return the imaginary part */
double d_imag(doublecomplex *z)
{
    return (z->i);
}
