/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>


#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3

#define SEND 0
#define RECV 1

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

typedef unsigned int uint;

typedef uint    vec2_t[2];  // vec2_t is an alias for array of 2 els of uints

typedef double *restrict buffers_t[4];

typedef struct {
    double   * restrict data;
    vec2_t     size;
} plane_t;  // this creates a plane object that has nx, ny, and an array of doubles (values)
            // row-major or col-major only depends on data initialization:
            // row-major: data[y * width + x]
            // col-major: data[x * height + y]

extern int inject_energy ( const int      ,
                           const int      ,
			               const vec2_t  *,
			               const double   ,
                                 plane_t *,
                           const vec2_t   );


extern int update_plane ( const int      ,
                          const vec2_t   ,
                          const plane_t *,
                                plane_t *);


extern int get_total_energy( plane_t *,
                             double  * );

extern int update_plane_border( const int               ,
                                const vec2_t            ,
                                const plane_t * restrict,
                                      plane_t * restrict);

extern int update_plane_interior( const plane_t * restrict,
                                        plane_t * restrict);

int initialize ( MPI_Comm *,
                 int       ,
		         int       ,
		         int       ,
		         char    **,
                 vec2_t   *,
                 vec2_t   *,                 
		         int      *,
                 int      *,
		         uint     *,
		         int      *,
		         int      *,
		         int      *,
                 vec2_t  **,
                 double   *,
                 plane_t  *,
                 buffers_t*, 
                 int      *);


int memory_release ( plane_t   * , 
                     buffers_t *);


int output_energy_stat ( int      ,
                         plane_t *,
                         double   ,
                         int      ,
                         MPI_Comm *);



inline int inject_energy ( const int      periodic,
                           const int      Nsources,
                           const vec2_t  *Sources,
                           const double   energy,
                                 plane_t *plane,
                           const vec2_t   N)
{
    const uint register sizex = plane->size[_x_] + 2;
    const uint register ysize = plane->size[_y_];
    double * restrict data = plane->data;

    #define IDX(i, j) ((j) * sizex + (i))

    for (int s = 0; s < Nsources; s++) {
        const int x = Sources[s][_x_];
        const int y = Sources[s][_y_];

        data[IDX(x, y)] += energy;

        if (periodic) {
            // handle periodic if there's only one MPI rank along X (vertical rectangle)
            if (N[_x_] == 1) {
                if (x == 1)
                    data[IDX(sizex - 1, y)] += energy;  // wrap right halo
                else if (x == sizex - 2)
                    data[IDX(0, y)] += energy;          // wrap left halo
            }

            // handle periodic if there's only one MPI rank along Y (horizontal rectangle)
            if (N[_y_] == 1) {
                if (y == 1)
                    data[IDX(x, ysize + 1)] += energy;  // wrap bottom halo
                else if (y == ysize)
                    data[IDX(x, 0)] += energy;          // wrap top halo
            }
        }
    }

    #undef IDX
return 0;
}



inline int update_plane_interior( const plane_t * restrict oldplane,
                                        plane_t * restrict newplane )
{
    uint register fxsize = oldplane->size[_x_] + 2;
    uint register xsize = oldplane->size[_x_];
    uint register ysize = oldplane->size[_y_];

    const double * restrict old = oldplane->data;
    double * restrict new = newplane->data;

    const double alpha = 0.6;
    const double scale = 0.25 * (1-alpha);

    // stencil update only inside the new plane
    #pragma omp parallel for schedule(static)
    for (int j = 2; j <= ysize - 1; ++j) {
        const double *row_center = old + j * fxsize;
        const double *row_up     = old + (j - 1) * fxsize;
        const double *row_down   = old + (j + 1) * fxsize;
        double *new_row          = new + j * fxsize;
        // assuming we want the center point to weight 0.6, and the other
        // points together (1 - 0.6), but since we have 4 points we have to
        // divide the 0.4 into 4, so each weights 0.1
        for (int i = 2; i <= xsize - 1; ++i) {
            new_row[i] = row_center[i] * 0.6  +  // itself
                        (row_center[i - 1]    +  // west
                         row_center[i + 1]    +  // east
                         row_up[i]            +  // north
                         row_down[i]) * 0.1;     // south
        }
    }
}

inline int update_plane_border( const int periodic,
                                const vec2_t N,
                                const plane_t *restrict oldplane,
                                      plane_t *restrict newplane )
{
    const int register fxsize = oldplane->size[_x_] + 2;
    const int register ysize  = oldplane->size[_y_];

    const double *restrict old = oldplane->data;
    double *restrict new       = newplane->data;

    #pragma omp parallel for schedule(static)
    for (int i = 1; i <= fxsize-2; ++i) {
        // update top row (j=1)
        new[fxsize + i] = old[fxsize + i] * 0.6 +       // itself
                         (old[fxsize + (i - 1)]   +     // east (1 halo)
                          old[fxsize + (i + 1)]   +     // west (1 halo)
                          old[i]                  +     // north (always halo)
                          old[2 * fxsize + i]) * 0.1;   // south

        // update bottom row (j=ysize)
        new[ysize * fxsize + i] = old[ysize * fxsize + i] * 0.6 +       // itself
                                 (old[ysize * fxsize + (i - 1)]   +       // east (1 halo)
                                  old[ysize * fxsize + (i + 1)]   +       // west (1 halo)
                                  old[(ysize - 1) * fxsize + i]   +       // north
                                  old[(ysize + 1) * fxsize + i]) * 0.1; // south (always halo)
    }

    #pragma omp parallel for schedule(static)
    for (int j = 2; j <= ysize - 1; ++j) { // first and last values were alredy updated
        // update left column (i=1)
        new[j * fxsize + 1] = old[j * fxsize + 1] * 0.6 +         // itself
                             (old[j * fxsize]             +       // west (always halo)
                              old[j * fxsize + 2]         +       // east
                              old[(j - 1) * fxsize + 1]   +       // north
                              old[(j + 1) * fxsize + 1]) * 0.1;   // south

        // update right column (i=xsize)
        new[(j+1) * fxsize - 2] = old[(j+1) * fxsize - 2] * 0.6 +   // itself
                                 (old[(j+1) * fxsize - 3] +         // west
                                  old[(j+1) * fxsize - 1] +         // east (always halo)
                                  old[(j) * fxsize - 2] +           // north
                                  old[(j+2) * fxsize -2]) * 0.1;    // south
    }

    if (periodic) {
        // handle periodic if there's only one MPI rank along X (vertical rectangle)
        if (N[_x_] == 1) {
            for (int j = 1; j <= ysize; ++j) {
                new[j * fxsize]           = new[(j+1) * fxsize -2];   // left halo = right edge
                new[(j+1) * fxsize -1] = new[j * fxsize + 1];       // right halo = left edge
            }
        }
        // handle periodic if there's only one MPI rank along Y (horizontal rectangle)
        if (N[_y_] == 1) {
            for (int i = 1; i <= fxsize-2; ++i) {
                new[i]                    = new[ysize * fxsize + i];   // top halo = bottom edge
                new[(ysize+1) * fxsize+i] = new[1 * fxsize + i];       // bottom halo = top edge
            }
        }
    }

    return 0;
}


inline int get_total_energy( plane_t *plane, 
                             double  *energy )
{
    const int register xsize = plane->size[_x_];
    const int register ysize = plane->size[_y_];
    const int register fxsize = xsize + 2;

    double * restrict data = plane->data;
    double totenergy = 0.0;

    #define IDX( i, j ) ( (j)*fxsize + (i) )

    for ( int j = 1; j <= ysize; j++ )
        for ( int i = 1; i <= xsize; i++ )
            totenergy += data[ IDX(i, j) ];
    
    
    #undef IDX
    
    *energy = (double)totenergy;
    return 0;
}




