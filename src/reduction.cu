//	INCLUDES
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>        	/* errno */
#include <string.h>       	/* strerror */
#include <math.h>			// ceil
#include <time.h>			// CLOCKS_PER_SEC

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// GIS
#include "/home/giuliano/git/cuda/weatherprog-cudac/includes/gis.h"

/**
 * 	PARS
 */
#define 		TILE_DIM_small				16
#define 		TILE_DIM 					32
bool 			print_intermediate_arrays 	= false;
const char 		*BASE_PATH 					= "/home/giuliano/git/cuda/reduction";

/*
 *	kernel labels
 */
const char 		*kern_1 		= "reduction"	;
/*const char 		*kern_2 		= "sum_of_3_cols"		;
const char 		*kern_3 		= "cumsum_vertical"		;
const char 		*kern_4 		= "sum_of_3_rows"		;
*/
char			buffer[255];

/*
 * 		DEFINE I/O files
 */
const char 		*FIL_ROI 		= "/home/giuliano/git/cuda/redution/data/ROI.tif";
const char 		*FIL_BIN1 		= "/home/giuliano/git/cuda/reduction/data/BIN1.tif";
const char 		*FIL_BIN2 		= "/home/giuliano/git/cuda/reduction/data/BIN2.tif";
const char 		*FIL_LTAKE_grid	= "/home/giuliano/git/cuda/reduction/data/FRAG-cuda.tif";
const char 		*FIL_LTAKE_num	= "/home/giuliano/git/cuda/reduction/data/FRAGt-cuda.tif";

/*	+++++DEFINEs+++++	*/
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
/*	+++++DEFINEs+++++	*/

template <typename T>
void write_mat_T( T *MAT, unsigned int nr, unsigned int nc, const char *filename )
{
	unsigned int rr,cc;
	FILE *fid ;
	fid = fopen(filename,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",filename); exit(1); }

	for(rr=0;rr<nr;rr++)
	{
		for(cc=0;cc<nc;cc++)
		{
			fprintf(fid, "%6.2f ",MAT[rr*nc+cc]);
		}
		fprintf(fid,"\n");
	}
	fclose(fid);
}


__global__ void imperviousness_change_sh4(
							const char 		*dev_BIN1,			const char		*dev_BIN2,
							unsigned int 	WIDTH,				unsigned int 	HEIGHT,
							unsigned int 	*dev_LTAKE_count,	int 			*dev_LTAKE_map )
{
/*
 INPUTS:
    dev_BIN1:			Imperviousness of OLDER year.
    dev_BIN1:			Imperviousness of NEWER year.
 OUTPUTS:
    dev_LTAKE_count:	2x2 table with counts about the 4 possible combinations.
    dev_LTAKE_map:      map storing the difference (dev_BIN2-dev_BIN1).
    					The following 4 combinations are possible computing the difference:
    						----------------
    						(N) (BIN2,BIN1)		---> 0:rural __ 1:urban
    						----------------
    						(1) (0,0) --> +0	---> nothing changed in rural pixels
    						(2) (0,1) --> -1	---> increase of urban pixels
    						(3) (1,0) --> +1	---> increase of rural pixels
    						(4) (1,1) --> -0	---> nothing changed in urban pixels
    						----------------
*/

	unsigned int x 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bix		= blockIdx.x;

	unsigned int tix		= bdx*bix + x;	// offset

	__shared__ int tmp[4];

	if( tix<WIDTH*HEIGHT ){
		if (tix<4) tmp[tix] = 0;

		dev_LTAKE_map[tix]	= dev_BIN2[tix] - dev_BIN1[tix];

		atomicAdd( &tmp[dev_BIN2[tix]*2+dev_BIN1[tix]], 1 );
		syncthreads();

		if (tix<4) atomicAdd( &dev_LTAKE_count[tix], tmp[tix] );
	}
}

int main( int argc, char **argv ){

	/*
	 * 	NOTES:
	 *
	 */

	/*
	 * 		ESTABILISH CONTEXT
	 */
	GDALAllRegister();	// Establish GDAL context.
	cudaFree(0); 		// Establish CUDA context.

	metadata 			MDbin,MDroi,MDdouble; // ,MDtranspose
	unsigned int		map_len;
	int 				*dev_LTAKE_map,*host_LTAKE_map;
	int 				*dev_LTAKE_count,*host_LTAKE_count;
	unsigned char		*dev_BIN, *dev_ROI;
	clock_t				start_t,end_t;
	unsigned int 		elapsed_time	= 0;
	cudaDeviceProp		devProp;
	unsigned int		gpuDev=0;
	// count the number of kernels that must print their output:
	unsigned int 		count_print = 0;

	// query current GPU properties:
	CUDA_CHECK_RETURN( cudaSetDevice(gpuDev) );
	cudaGetDeviceProperties(&devProp, gpuDev);

	/*
	 * 		LOAD METADATA & DATA
	 */
	MDbin					= geotiffinfo( FIL_BIN1, 1 );
	MDroi 					= geotiffinfo( FIL_ROI, 1 );
	// set metadata to eventually print arrays after any CUDA kernel:
	MDdouble 				= MDbin;
	MDdouble.pixel_type		= GDT_Float64;
	// Set size of all arrays which come into play:
	map_len 				= MDbin.width*MDbin.heigth;
	size_t	sizeChar		= map_len*sizeof( unsigned char );
	size_t	sizeInt			= map_len*sizeof( int );
	// initialize arrays:
	unsigned char *BIN		= (unsigned char *) CPLMalloc( sizeChar );
	unsigned char *ROI 		= (unsigned char *) CPLMalloc( sizeChar );
	// load ROI:
	printf("Importing...\t%s\n",FIL_ROI);
	geotiffread( FIL_ROI, MDroi, &ROI[0] );
	// load BIN:
	printf("Importing...\t%s\n",FIL_BIN1);
	geotiffread( FIL_BIN1, MDbin, &BIN[0] );

	/*
	 * 	INITIALIZE CPU & GPU ARRAYS
	 */
	// initialize grids on CPU MEM:
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_LTAKE_map, 	sizeInt)  );
	// initialize grids on GPU MEM:
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_BIN, 			sizeChar) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_ROI,  		sizeChar) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_LTAKE_map, 	sizeInt)  );
	// memset:
/*	CUDA_CHECK_RETURN( cudaMemset(dev_ROI, 0,  					sizeInt) );
	CUDA_CHECK_RETURN( cudaMemset(dev_BIN, 0,  					sizeInt) );
*/	// H2D:
	CUDA_CHECK_RETURN( cudaMemcpy(dev_BIN, BIN, 	sizeChar, cudaMemcpyHostToDevice) );
	CUDA_CHECK_RETURN( cudaMemcpy(dev_ROI, ROI, 	sizeChar, cudaMemcpyHostToDevice) );

	/*
	 * 		KERNELS GEOMETRY
	 * 		NOTE: use ceil() instead of the "%" operator!!!
	 */
	int sqrt_nmax_threads = TILE_DIM;//floor(sqrt( devProp.maxThreadsPerBlock ));
	// k1 + k2
	unsigned int 	gdx_k12;
	gdx_k 			= ((unsigned int)(MDbin.width % mask_len)>0) + (MDbin.width  / mask_len);
	gdy_k 			= (unsigned int)(MDbin.heigth % (sqrt_nmax_threads*sqrt_nmax_threads)>0) + floor(MDbin.heigth / (sqrt_nmax_threads*sqrt_nmax_threads));
	dim3 block_k( 1,sqrt_nmax_threads*sqrt_nmax_threads,1);
	dim3 grid_k ( gdx_k12,gdy_k12,1);

	/*		KERNELS INVOCATION
	 *
	 *			*************************
	 *			-1- ??
	 *			-2- ??
	 *			*************************
	 */
	printf("\n\n");
	// ***-1-***
	start_t = clock();
	reduction<<<grid_k,block_k>>>( 	dev_BIN, dev_ROI, MDbin.width, MDbin.heigth,  	);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %12s\t%6d [msec]\n",++count_print,kern_1,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(host_IO,dev_IO,	(size_t)sizeInt,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_1);
		geotiffwrite( FIL_BIN1, buffer, MDdouble, host_IO );
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

	printf("______________________________________\n");
	printf("  %16s\t%6d [msec]\n", "Total time:",elapsed_time );

	CUDA_CHECK_RETURN( cudaMemcpy(host_LTAKE_map,dev_LTAKE_map,	(size_t)sizeInt,cudaMemcpyDeviceToHost) );
	// save on HDD
	geotiffwrite( FIL_FRAG, FIL_FRAG_2, MDdouble, host_LTAKE );

	// CUDA free:
	cudaFree( dev_BIN	);
	cudaFree( dev_LTAKE_map	);
	cudaFree( dev_ROI	);

	// Destroy context
	CUDA_CHECK_RETURN( cudaDeviceReset() );

	printf("\n\n\nFinished!!\n");

	return 0;// elapsed_time
}
