# parallel-mandelbrot-set
Parallelize sequential mandelbrot set using pthread and MPI+openMP.  
Rank (based on execution time):  
- pthread version (8/65)
- MPI + openMP version (3/63)

# I/O spec
`srun -n $procs -c $t ./hw2a $out $iter $x0 $x1 $y0 $y1 $w $h`

`$procs` (int): number of processes (1 in pthread version).  
`$t` (int): number of threads per process.  
`$out` (string): the path to the output file.  
`$iter` (int): number of iterations.  
`$x0` (double): inclusive bound of the real axis.  
`$x1` (double): non-inclusive bound of the real axis.  
`$y0` (double): inclusive bound of the imaginary axis.  
`$y1` (double): non-inclusive bound of the imaginary axis.  
`$w` (int): number of points in the x-axis for output.  
`$h` (int): number of points in the y-axis for output.  

# Implementation
## pthread version
used vectorization (SSE2), full details in the report.

## MPI + OpenMP 
details in the report
