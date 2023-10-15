#include <iostream>
#include <complex>
#include <omp.h>
#include <string>

#include "util.h"

float byteToMB(long bytes) {
    return (float)bytes/(1024.0*1024.0);
}	

/************************************************************************************/
/* Re-order an xGPU triangular order visibility set to MWAX order                   */
/* It's hard to explain my algorithm - it's based on visual inspcection of xGPU's   */
/* output triangle versus what we want for MWAX order - including swapping from     */
/* A vs B to B vs A.                                                                */
/************************************************************************************/
// "MWAX order" is [time][baseline][channel][polarisation]
// can rearrange both xGPU triangular and TCC triangular baseline orderings
void tri_to_mwax(Parameters params, std::complex<float> *tri_buffer, std::complex<float> *mwax_buffer_in)
{
    int i, j, f;

    int num_tiles = params.nstation;
    
    int column_length = num_tiles;
    int first_baseline_step = 1;
    int baseline_step = first_baseline_step;
    int first_baseline_index = 0;
    int baseline_index = first_baseline_index;
    int visibility_index;

    float *mwax_buffer = (float*)mwax_buffer_in;

    for (i=0; i<num_tiles; i++)  // all columns of xGPU's upper triangle
    {
        for (j=0; j<column_length; j++)   // number of rows in each output column decrements (lower triangle) (j=0 is the auto)
        {
            for (f=0; f<params.nfrequency; f++)
            {
                // visibility_index = 4*baseline_index + f*ctx->num_xgpu_visibilities_per_chan;  // x4 for 4 xpols
                // comes out of TCC as [baseline][channel][polarisation][polarisation]
                visibility_index = baseline_index*params.nfrequency*params.npol*params.npol + f*params.npol*params.npol;

                // conjugate and swap xy/yx because swapping tile A / tile B in going from xGPU triangular format to MWAX format
                *mwax_buffer++ = std::real(tri_buffer[visibility_index]);      // xx real
                *mwax_buffer++ = -std::imag(tri_buffer[visibility_index]);     // xx imag - conjugate
                *mwax_buffer++ = std::real(tri_buffer[visibility_index + 2]);  // yx real
                *mwax_buffer++ = -std::imag(tri_buffer[visibility_index + 2]); // yx imag - conjugate
                *mwax_buffer++ = std::real(tri_buffer[visibility_index + 1]);  // xy real
                *mwax_buffer++ = -std::imag(tri_buffer[visibility_index + 1]); // xy imag - conjugate
                *mwax_buffer++ = std::real(tri_buffer[visibility_index + 3]);  // yy real
                *mwax_buffer++ = -std::imag(tri_buffer[visibility_index + 3]); // yy imag - conjugate
            }
            baseline_index += baseline_step;
            baseline_step++;
        }
        column_length--;
        first_baseline_step++;
        baseline_step = first_baseline_step;
        first_baseline_index += first_baseline_step;
        baseline_index = first_baseline_index;
    }
    return;
}