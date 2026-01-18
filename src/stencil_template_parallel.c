/*

/*
 *
 *  mysizex   :   local x-extendion of your patch
 *  mysizey   :   local y-extension of your patch
 *
 */


#include "../include/stencil_template_parallel.h"



// ------------------------------------------------------------------
// ------------------------------------------------------------------

int main(int argc, char **argv)
{
  MPI_Comm myCOMM_WORLD;
  int  Rank, Ntasks;
  int  neighbours[4];

  int  Niterations;
  int  periodic;
  vec2_t S, N;
  
  int      Nsources;
  int      Nsources_local;
  vec2_t  *Sources_local;
  double   energy_per_source;

  plane_t   planes[2];  
  buffers_t buffers[2];

  int output_energy_stat_perstep;
  int verbose;


  // INITIALIZE
  {
    int level_obtained;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &level_obtained);
    if (level_obtained < MPI_THREAD_FUNNELED) {
      printf("MPI_thread level obtained is %d instead of %d\n",level_obtained, MPI_THREAD_FUNNELED );
      MPI_Finalize();
      exit(1);
    }
    
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    MPI_Comm_size(MPI_COMM_WORLD, &Ntasks);
    MPI_Comm_dup(MPI_COMM_WORLD, &myCOMM_WORLD);
  }

  int ret = initialize(&myCOMM_WORLD, Rank, Ntasks, argc, argv,
                       &S, &N, &periodic, &output_energy_stat_perstep,
                       neighbours, &Niterations,
                       &Nsources, &Nsources_local, &Sources_local,
                       &energy_per_source,
                       &planes[0], &buffers[0], &verbose);

  if (ret) {
    printf("Rank %d: initialization failed with code %d\n", Rank, ret); 
    MPI_Finalize();
    return 1;
  }

  // TIMERS
  double t_start, t_end;
  double t_communication = 0.0, t_computation = 0.0;
  double t_comp, t_comm; // used to set the start of each phase

  int current = OLD;

  // MAIN ITERATION LOOP
  t_start = MPI_Wtime();

  for (int iter = 0; iter < Niterations; ++iter)
  {
    MPI_Request reqs[8];
    int rq = 0;

    // INJECT ENERGY
    inject_energy(periodic, Nsources_local, Sources_local,
                  energy_per_source, &planes[current], N);

    // BUFFER PACKING
    double *old = planes[current].data;
    uint nx = planes[current].size[_x_];
    uint ny = planes[current].size[_y_];
    uint fx = nx + 2;

    t_comm = MPI_Wtime();

    if (neighbours[NORTH] != MPI_PROC_NULL){
      buffers[SEND][NORTH] = &old[fx + 1];
      buffers[RECV][NORTH] = &old[1];
    } else {
      buffers[SEND][NORTH] = buffers[RECV][NORTH] = NULL;
    }

    if (neighbours[SOUTH] != MPI_PROC_NULL){
      buffers[SEND][SOUTH] = &old[ny * fx + 1];
      buffers[RECV][SOUTH] = &old[(ny + 1) * fx + 1];
    } else {
      buffers[SEND][SOUTH] = buffers[RECV][SOUTH] = NULL;
    }

    if (neighbours[WEST] != MPI_PROC_NULL){
      for (uint j = 0; j < ny; ++j)
        buffers[SEND][WEST][j] = old[(j+1)*fx + 1];
    }
    if (neighbours[EAST] != MPI_PROC_NULL){
      for (uint j = 0; j < ny; ++j)
        buffers[SEND][EAST][j] = old[(j+1)*fx + nx];
    }
    
    // START THE COMUNICATION

    if (neighbours[NORTH] != MPI_PROC_NULL) {
      MPI_Irecv(buffers[RECV][NORTH], nx, MPI_DOUBLE,
                neighbours[NORTH], 100, myCOMM_WORLD, &reqs[rq++]);
      MPI_Isend(buffers[SEND][NORTH], nx, MPI_DOUBLE,
                neighbours[NORTH], 200, myCOMM_WORLD, &reqs[rq++]);
    }
    if (neighbours[SOUTH] != MPI_PROC_NULL) {
      MPI_Irecv(buffers[RECV][SOUTH], nx, MPI_DOUBLE,
                neighbours[SOUTH], 200, myCOMM_WORLD, &reqs[rq++]);
      MPI_Isend(buffers[SEND][SOUTH], nx, MPI_DOUBLE,
                neighbours[SOUTH], 100, myCOMM_WORLD, &reqs[rq++]);
    }
    if (neighbours[WEST] != MPI_PROC_NULL) {
      MPI_Irecv(buffers[RECV][WEST], ny, MPI_DOUBLE,
                neighbours[WEST], 300, myCOMM_WORLD, &reqs[rq++]);
      MPI_Isend(buffers[SEND][WEST], ny, MPI_DOUBLE,
                neighbours[WEST], 400, myCOMM_WORLD, &reqs[rq++]);
    }
    if (neighbours[EAST] != MPI_PROC_NULL) {
      MPI_Irecv(buffers[RECV][EAST], ny, MPI_DOUBLE,
                neighbours[EAST], 400, myCOMM_WORLD, &reqs[rq++]);
      MPI_Isend(buffers[SEND][EAST], ny, MPI_DOUBLE,
                neighbours[EAST], 300, myCOMM_WORLD, &reqs[rq++]);
    }
    t_communication += MPI_Wtime() - t_comm;

    // UPDATE THE INTERIOR OF THE PLANES (while communicating, which do not require halos)
    t_comp = MPI_Wtime();
    update_plane_interior(&planes[current], &planes[!current]);
    t_computation += MPI_Wtime() - t_comp;

    MPI_Waitall(rq, reqs, MPI_STATUSES_IGNORE);

    // UNPACK WEST AND EAST BUFFERS (after recieving all the halos, since north 
    // and south have alredy been copyied in memory)
    t_comp = MPI_Wtime();
    if (neighbours[EAST] != MPI_PROC_NULL)
    {
      for (uint j = 0; j < ny; ++j)
        old[(j + 1) * fx + (nx + 1)] = buffers[RECV][EAST][j];
    }
    if (neighbours[WEST] != MPI_PROC_NULL)
    {
      for (uint j = 0; j < ny; ++j)
        old[(j + 1) * fx] = buffers[RECV][WEST][j];
    }

    // UPDATE THE BORDERS (after waiting for all communications to end)
    update_plane_border(periodic, N, &planes[current], &planes[!current]);
    t_computation += MPI_Wtime() - t_comp;

    if (output_energy_stat_perstep) {
      output_energy_stat(iter, &planes[!current],
                         (iter+1) * Nsources * energy_per_source,
                         Rank, &myCOMM_WORLD);}

    current = !current;
  }
  t_end = MPI_Wtime() - t_start;

  // RELEASE MEMORY
  if (output_energy_stat_perstep) {
  output_energy_stat(-1, &planes[!current],
                     Niterations * Nsources * energy_per_source,
                     Rank, &myCOMM_WORLD); }

  ret = memory_release(planes, buffers);

  // REDUCE ALL TIMING

  // Local timers
  double local_comp  = t_computation;
  double local_comm = t_communication;
  double local_total   = t_computation + t_communication;

  // Max and avg reductions
  double max_comp,  max_comm,  max_total;
  double sum_comp,  sum_comm,  sum_total;

  MPI_Reduce(&local_comp,  &max_comp,   1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
  MPI_Reduce(&local_comm,  &max_comm,   1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);
  MPI_Reduce(&local_total, &max_total,  1, MPI_DOUBLE, MPI_MAX, 0, myCOMM_WORLD);

  MPI_Reduce(&local_comp,  &sum_comp,   1, MPI_DOUBLE, MPI_SUM, 0, myCOMM_WORLD);
  MPI_Reduce(&local_comm,  &sum_comm,   1, MPI_DOUBLE, MPI_SUM, 0, myCOMM_WORLD);
  MPI_Reduce(&local_total, &sum_total,  1, MPI_DOUBLE, MPI_SUM, 0, myCOMM_WORLD);

  if (Rank == 0) {
    int n_ranks = 0;
    MPI_Comm_size(myCOMM_WORLD, &n_ranks);

    // determine number of nodes
    int n_nodes = 1;
    char *env_nnodes = getenv("SLURM_NNODES");
    if (env_nnodes != NULL) {
      n_nodes = atoi(env_nnodes);
      if (n_nodes <= 0) n_nodes = 1;
    }

    // OpenMP threads
    int n_threads = 1;
    #ifdef _OPENMP
        n_threads = omp_get_max_threads();
    #endif

    double avg_comp = sum_comp / (double)n_ranks;
    double avg_comm = sum_comm / (double)n_ranks;
    double avg_total = sum_total / (double)n_ranks;

    // CSV: NODES,MPI_TASKS,OMP_THREADS,GRID_X,GRID_Y,MAX_TOTAL,
    //      AVG_TOTAL,MAX_COMP,AVG_COMP,MAX_COMM,AVG_COMM
    printf("%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
          n_nodes, n_ranks, n_threads,
          planes[!current].size[_x_], planes[!current].size[_y_],
          max_total, avg_total,
          max_comp, avg_comp,
          max_comm, avg_comm);
  
}


  MPI_Finalize();
  return 0;

}

/* ==========================================================================
   =                                                                        =
   =   routines called within the integration loop                          =
   ========================================================================== */





/* ==========================================================================
   =                                                                        =
   =   initialization                                                       =
   ========================================================================== */


uint simple_factorization( uint, int *, uint ** );

int initialize_sources( int       ,
			int       ,
			MPI_Comm  *,
			uint      [2],
			int       ,
			int      *,
			vec2_t  ** );


int memory_allocate ( const int       *,
		      const vec2_t     ,
		            buffers_t *,
		            plane_t   * );
		        
int initialize ( 
    MPI_Comm  *Comm,
		int        Me,                  // the rank of the calling process
		int        Ntasks,              // the total number of MPI ranks
		int        argc,                // the argc from command line
		char     **argv,                // the argv from command line
		vec2_t    *S,                   // the size of the plane
		vec2_t    *N,                   // two-uint array defining the MPI tasks' grid
		int       *periodic,            // periodic-boundary tag
		int       *output_energy_stat_perstep,
		uint      *neighbours,          // four-int array that gives back the neighbours of the calling task
		int       *Niterations,         // how many iterations
		int       *Nsources,            // how many heat sources
		int       *Nsources_local,
		vec2_t   **Sources_local,
		double    *energy_per_source,   // how much heat per source
		plane_t   *planes,
		buffers_t *buffers,
    int       *verbose
		)
{
  int halt = 0;
  int ret;
  
  // ··································································
  // set default values

  (*S)[_x_]         = 100;
  (*S)[_y_]         = 100;
  *periodic         = 1;
  *Nsources         = 10;
  *Nsources_local   = 0;
  *Sources_local    = NULL;
  *Niterations      = 100;
  *energy_per_source = 1.0;
  *verbose = 0;
  *output_energy_stat_perstep = 0;

  planes[OLD].size[0] = planes[OLD].size[1] = 0;
  planes[NEW].size[0] = planes[NEW].size[1] = 0;
  
  for ( int i = 0; i < 4; i++ )
    neighbours[i] = MPI_PROC_NULL;

  for ( int b = 0; b < 2; b++ )
    for ( int d = 0; d < 4; d++ )
      buffers[b][d] = NULL;

  // ··································································
  // process the command line
  // 
  while ( 1 )
  {
    int opt;
    while((opt = getopt(argc, argv, ":h:x:y:e:E:n:o:p:v:")) != -1)
      {
	switch( opt )
	  {
	  case 'x': (*S)[_x_] = (uint)atoi(optarg);
	    break;

	  case 'y': (*S)[_y_] = (uint)atoi(optarg);
	    break;

	  case 'e': *Nsources = atoi(optarg);
	    break;

	  case 'E': *energy_per_source = atof(optarg);
	    break;

	  case 'n': *Niterations = atoi(optarg);
	    break;

	  case 'v': *verbose = (atoi(optarg) > 0);
	    break;

	  case 'p': *periodic = (atoi(optarg) > 0);
	    break;
  
    case 'o': *output_energy_stat_perstep = (atoi(optarg) > 0);
	    break;

	  case 'h': {
	    if ( Me == 0 )
	      printf( "\nvalid options are ( values btw [] are the default values ):\n"
		      "-x    x size of the plate [10000]\n"
		      "-y    y size of the plate [10000]\n"
		      "-e    how many energy sources on the plate [4]\n"
		      "-E    how much energy each source pumps [1.0]\n"
		      "-n    how many iterations [1000]\n"
		      "-v    whether to print a verbose output [0]\n"
		      "-p    whether periodic boundaries applies  [0 = false]\n"
		      "-o    whether to output energy at each step  [0 = false]\n"
		      );
	    halt = 1; }
	    break;
	    
	    
	  case ':': printf( "option -%c requires an argument\n", optopt);
	    break;
	    
	  case '?': printf(" -------- help unavailable ----------\n");
	    break;
	  }
      }

    if ( opt == -1 )
      break;
  }

  if ( halt )
    return 1;
  
  
  // ··································································
  /*
   * here we check for all the parms being meaningful
   *
   */
  if ((*S)[_x_] < 100) {
    fprintf(stderr, "Warning: grid x-dimension too small (%u), forcing to 100\n", (*S)[_x_]);
    (*S)[_x_] = 100;
  }

  if ((*S)[_y_] < 100) {
      fprintf(stderr, "Warning: grid y-dimension too small (%u), forcing to 100\n", (*S)[_y_]);
      (*S)[_y_] = 100;
  }

  if (*Nsources <= 0) {
    fprintf(stderr,
        "Warning: Nsources was %d, forcing to 1 (at least one source required)\n",
        *Nsources);
    *Nsources = 1;
  }

  if (*energy_per_source <= 0) {
    fprintf(stderr,
        "Warning: energy_per_source was %f, forcing to 1.0\n",
        *energy_per_source);
    *energy_per_source = 1.0;
  }

  // ··································································
  /*
   * find a suitable domain decomposition
   * very simple algorithm, you may want to
   * substitute it with a better one
   *
   * the plane Sx x Sy will be solved with a grid
   * of Nx x Ny MPI tasks
   * 
   *  MAYBE TRY WITH THIS AND THEN WHEN IT WORKS COMPARE WITH ANOTHER ALGO AND TAKE BEST
   *  FOR NOW THIS WORKS
   */

  vec2_t Grid; // how many procs per coordinate
  double formfactor = ((*S)[_x_] >= (*S)[_y_] ? (double)(*S)[_x_]/(*S)[_y_] : (double)(*S)[_y_]/(*S)[_x_] );
  int    dimensions = 2 - (Ntasks <= ((int)formfactor+1) ); // truncates down to int

  
  if ( dimensions == 1 )
    {
      if ( (*S)[_x_] >= (*S)[_y_] )
	Grid[_x_] = Ntasks, Grid[_y_] = 1;
      else
	Grid[_x_] = 1, Grid[_y_] = Ntasks;
    }
  else
    {
      int   Nf;
      uint *factors;
      uint  first = 1;
      ret = simple_factorization( Ntasks, &Nf, &factors );
      
      for ( int i = 0; (i < Nf) && ((Ntasks/first)/first > formfactor); i++ )
	first *= factors[i];

      if ( (*S)[_x_] > (*S)[_y_] )
	Grid[_x_] = Ntasks/first, Grid[_y_] = first;
      else
	Grid[_x_] = first, Grid[_y_] = Ntasks/first;
    }

  (*N)[_x_] = Grid[_x_];
  (*N)[_y_] = Grid[_y_];
  

  // ··································································
  // my coordinates in the grid of processors
  //
  int X = Me % Grid[_x_];  // instead of the 1D ranks we transform into 2D ranks of MPI procs
  int Y = Me / Grid[_x_];

  // ··································································
  // find my neighbours
  //

  if ( Grid[_x_] > 1 )
    {  
      if ( *periodic ) {       
	neighbours[EAST]  = Y*Grid[_x_] + (Me + 1 ) % Grid[_x_];
	neighbours[WEST]  = (X%Grid[_x_] > 0 ? Me-1 : (Y+1)*Grid[_x_]-1); }
      
      else {
	neighbours[EAST]  = ( X < Grid[_x_]-1 ? Me+1 : MPI_PROC_NULL );
	neighbours[WEST]  = ( X > 0 ? (Me-1)%Ntasks : MPI_PROC_NULL ); }  
    }

  if ( Grid[_y_] > 1 )
    {
      if ( *periodic ) {      
	neighbours[NORTH] = (Ntasks + Me - Grid[_x_]) % Ntasks;
	neighbours[SOUTH] = (Ntasks + Me + Grid[_x_]) % Ntasks; }

      else {    
	neighbours[NORTH] = ( Y > 0 ? Me - Grid[_x_]: MPI_PROC_NULL );
	neighbours[SOUTH] = ( Y < Grid[_y_]-1 ? Me + Grid[_x_] : MPI_PROC_NULL ); }
    }

  // ··································································
  // the size of my patch
  //

  /*
   * every MPI task determines the size sx x sy of its own domain
   * REMIND: the computational domain will be embedded into a frame
   *         that is (sx+2) x (sy+2)
   *         the outern frame will be used for halo communication
   */
  
  vec2_t mysize;
  uint s = (*S)[_x_] / Grid[_x_];
  uint r = (*S)[_x_] % Grid[_x_];
  mysize[_x_] = s + (X < r);
  s = (*S)[_y_] / Grid[_y_];
  r = (*S)[_y_] % Grid[_y_];
  mysize[_y_] = s + (Y < r);

  planes[OLD].size[0] = mysize[0];
  planes[OLD].size[1] = mysize[1];
  planes[NEW].size[0] = mysize[0];
  planes[NEW].size[1] = mysize[1];
  

  if (*output_energy_stat_perstep)
    {
      if ( Me == 0 ) {
	  printf("Tasks are decomposed in a grid %d x %d\n\n", Grid[_x_], Grid[_y_] );
	  fflush(stdout);

      }

      MPI_Barrier(*Comm);
      
      for ( int t = 0; t < Ntasks; t++ )
	{
	  if ( t == Me )
	    {
	      printf("Task %4d :: "
		    "\tgrid coordinates : %3d, %3d\n"
		    "\tneighbours: N %4d    E %4d    S %4d    W %4d\n",
		    Me, X, Y,
		    neighbours[NORTH], neighbours[EAST],
		    neighbours[SOUTH], neighbours[WEST] );
	      fflush(stdout);

	    }

	  MPI_Barrier(*Comm);
	}
      
    }

  
  // ··································································
  // allocate the needed memory
  //
  ret = memory_allocate(neighbours, mysize, buffers, planes);
  if (ret)
    return ret;

  // ··································································
  // allocate the heat sources
  //
  ret = initialize_sources( Me, Ntasks, Comm, mysize, *Nsources, Nsources_local, Sources_local );
  
  return 0;  
}


uint simple_factorization( uint A, int *Nfactors, uint **factors )
/*
 * rought factorization;
 * assumes that A is small, of the order of <~ 10^5 max,
 * since it represents the number of tasks
 #
 */
{
  int N = 0;
  int f = 2;
  uint _A_ = A;

  while ( f < A )
    {
      while( _A_ % f == 0 ) {
	N++;
	_A_ /= f; }

      f++;
    }

  *Nfactors = N;
  uint *_factors_ = (uint*)malloc( N * sizeof(uint) );

  N   = 0;
  f   = 2;
  _A_ = A;

  while ( f < A )
    {
      while( _A_ % f == 0 ) {
	_factors_[N++] = f;
	_A_ /= f; }
      f++;
    }

  *factors = _factors_;
  return 0;
}


int initialize_sources( int       Me,
			int       Ntasks,
			MPI_Comm *Comm,
			vec2_t    mysize,
			int       Nsources,
			int      *Nsources_local,
			vec2_t  **Sources )

{

  srand48(time(NULL) ^ Me);
  int *tasks_with_sources = (int*)malloc( Nsources * sizeof(int) );
  
  if ( Me == 0 )
    {
      for ( int i = 0; i < Nsources; i++ )
	tasks_with_sources[i] = (int)lrand48() % Ntasks;
    }
  
  MPI_Bcast( tasks_with_sources, Nsources, MPI_INT, 0, *Comm );

  int nlocal = 0;
  for ( int i = 0; i < Nsources; i++ )
    nlocal += (tasks_with_sources[i] == Me);
  *Nsources_local = nlocal;
  
  if ( nlocal > 0 )
    {
      vec2_t * restrict helper = (vec2_t*)malloc( nlocal * sizeof(vec2_t) );      
      for ( int s = 0; s < nlocal; s++ )
	{
	  helper[s][_x_] = 1 + lrand48() % mysize[_x_];
	  helper[s][_y_] = 1 + lrand48() % mysize[_y_];
	}

      *Sources = helper;
    }
  
  free( tasks_with_sources );

  return 0;
}


int memory_allocate ( const int       *neighbours,
                      const vec2_t     N,
                            buffers_t *buffers_ptr,
                            plane_t   *planes_ptr )
{
  if (planes_ptr == NULL || buffers_ptr == NULL)
    return 1; // invalid pointers

  const int sizex = N[_x_];
  const int sizey = N[_y_];
  const int sizex_h = sizex + 2;
  uint frame_size = (sizex+2) * (sizey+2);

  // allocate memory for the OLD and NEW planes
  planes_ptr[OLD].data = calloc((size_t)frame_size, sizeof(double));
  planes_ptr[NEW].data = calloc((size_t)frame_size, sizeof(double));

  if (!planes_ptr[OLD].data || !planes_ptr[NEW].data)
    return 2; // invalid plane memory allocation

  // allocate send/recv buffers for each direction
  for (int b = 0; b < 2; b++)
    for (int d = 0; d < 4; d++)
      buffers_ptr[b][d] = NULL;

  // make north and south buffers point to plane memory
  if (neighbours[NORTH] != MPI_PROC_NULL)
  {
    buffers_ptr[SEND][NORTH] = &planes_ptr[OLD].data[1 * sizex_h + 1]; // interior first row
    buffers_ptr[RECV][NORTH] = &planes_ptr[OLD].data[0 * sizex_h + 1]; // north halo row
  }
  if (neighbours[SOUTH] != MPI_PROC_NULL)
  {
    buffers_ptr[SEND][SOUTH] = &planes_ptr[OLD].data[sizey * sizex_h + 1];       // interior last row
    buffers_ptr[RECV][SOUTH] = &planes_ptr[OLD].data[(sizey + 1) * sizex_h + 1]; // south halo row
  }

  // allocate memory for east buffer
  if (neighbours[EAST] != MPI_PROC_NULL)
  {
    buffers_ptr[SEND][EAST] = calloc((size_t)sizey, sizeof(double));
    buffers_ptr[RECV][EAST] = calloc((size_t)sizey, sizeof(double));
    
    if (!buffers_ptr[SEND][EAST] || !buffers_ptr[RECV][EAST])
    {
      if (buffers_ptr[SEND][EAST])
        free(buffers_ptr[SEND][EAST]);
      if (buffers_ptr[RECV][EAST])
        free(buffers_ptr[RECV][EAST]);
      free(planes_ptr[OLD].data);
      free(planes_ptr[NEW].data);
      return 3; // invalid east buffer memory allocation
    }
  }

  // allocate memory for west buffer
  if (neighbours[WEST] != MPI_PROC_NULL)
  {
    buffers_ptr[SEND][WEST] = calloc((size_t)sizey, sizeof(double));
    buffers_ptr[RECV][WEST] = calloc((size_t)sizey, sizeof(double));

    if (!buffers_ptr[SEND][WEST] || !buffers_ptr[RECV][WEST])
    {
      if (buffers_ptr[SEND][WEST])
        free(buffers_ptr[SEND][WEST]);
      if (buffers_ptr[RECV][WEST])
        free(buffers_ptr[RECV][WEST]);
      if (neighbours[EAST] != MPI_PROC_NULL)
      {
        free(buffers_ptr[SEND][EAST]);
        free(buffers_ptr[RECV][EAST]);
      }
      free(planes_ptr[OLD].data);
      free(planes_ptr[NEW].data);
      return 4; // invalid west buffer memory allocation
    }
  }

  return 0;
}


int memory_release(plane_t   *planes,
                   buffers_t *buffers)
{
  if (planes != NULL) {
    if (planes[OLD].data != NULL) {
      free(planes[OLD].data);
      planes[OLD].data = NULL;
    }
    if (planes[NEW].data != NULL) {
      free(planes[NEW].data);
      planes[NEW].data = NULL;
    }
  }

  // free east-west buffers
  if (buffers[SEND][EAST])
  {free(buffers[SEND][EAST]);
    buffers[SEND][EAST] = NULL;}

  if (buffers[RECV][EAST])
  {free(buffers[RECV][EAST]);
    buffers[RECV][EAST] = NULL;}

  if (buffers[SEND][WEST])
  {free(buffers[SEND][WEST]);
    buffers[SEND][WEST] = NULL;}

  if (buffers[RECV][WEST])
  {free(buffers[RECV][WEST]);
    buffers[RECV][WEST] = NULL;}

  // clear north-south pointers
  buffers[SEND][NORTH] = buffers[RECV][NORTH] = NULL;
  buffers[SEND][SOUTH] = buffers[RECV][SOUTH] = NULL;


  return 0;
}





int output_energy_stat ( int step, plane_t *plane, double budget, int Me, MPI_Comm *Comm )
{

  double system_energy = 0;
  double tot_system_energy = 0;
  get_total_energy ( plane, &system_energy );
  MPI_Reduce ( &system_energy, &tot_system_energy, 1, MPI_DOUBLE, MPI_SUM, 0, *Comm );
  
  if ( Me == 0 )
    {
      if ( step >= 0 )
	printf(" [ step %4d ] ", step ); 

      
      printf( "total injected energy is %g, "
	      "system energy is %g "
	      "( in avg %g per grid (%g , %g) point)\n",
	      budget,
	      tot_system_energy,
	      tot_system_energy / (plane->size[_x_]*plane->size[_y_]),
        plane->size[_x_],
        plane->size[_y_] );
    }
  
  return 0;
}