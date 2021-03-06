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
#define 					BLOCK_DIM_small				64
#define 					BLOCK_DIM 					256

static const unsigned int 	threads 					= 512;
bool 						print_intermediate_arrays 	= false;
const char 					*BASE_PATH 					= "/home/giuliano/git/cuda/reduction";

/*
 *	kernel labels
 */
const char		*kern_0			= "filter_roi";
const char 		*kern_1 		= "imperviousness_change_histc_sh_4"	;
const char 		*kern_2 		= "imperviousness_change"	;
char			buffer[255];

/*
 * 		DEFINE I/O files
 */
// I/–
//const char 		*FIL_ROI 		= "/home/giuliano/git/cuda/reduction/data/ROI.tif";
//const char 		*FIL_BIN1 		= "/home/giuliano/git/cuda/reduction/data/BIN1.tif";
//const char 		*FIL_BIN2 		= "/home/giuliano/git/cuda/reduction/data/BIN2.tif";
const char 		*FIL_ROI 		= "/media/DATI/db-backup/ssgci-data/testing/ssgci_roi.tif";
const char 		*FIL_BIN1 		= "/media/DATI/db-backup/ssgci-data/testing/ssgci_bin.tif";
const char 		*FIL_BIN2 		= "/media/DATI/db-backup/ssgci-data/testing/ssgci_bin2.tif";

// –/O
const char 		*FIL_LTAKE_grid	= "/home/giuliano/git/cuda/reduction/data/LTAKE_map.tif";
const char 		*FIL_LTAKE_count= "/home/giuliano/git/cuda/reduction/data/LTAKE_count.txt";

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
void write_mat_T( const T *MAT, unsigned int nr, unsigned int nc, const char *filename )
{
	unsigned int rr,cc;
	FILE *fid ;
	fid = fopen(filename,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",filename); exit(1); }

	for(rr=0;rr<nr;rr++)
	{
		for(cc=0;cc<nc;cc++)
		{
			fprintf(fid, "%8d ",MAT[rr*nc+cc]);
		}
		fprintf(fid,"\n");
	}
	fclose(fid);
}
void write_mat_int( const int *MAT, unsigned int nr, unsigned int nc, const char *filename )
{
	unsigned int rr,cc;
	FILE *fid ;
	fid = fopen(filename,"w");
	if (fid == NULL) { printf("Error opening file %s!\n",filename); exit(1); }

	for(rr=0;rr<nr;rr++)
	{
		for(cc=0;cc<nc;cc++)
		{
			fprintf(fid, "%8d ",MAT[rr*nc+cc]);
		}
		fprintf(fid,"\n");
	}
	fclose(fid);
}

__global__ void
filter_roi( unsigned char *BIN, const unsigned char *ROI, unsigned int map_len){
    unsigned int tid 		= threadIdx.x;
    unsigned int bix 		= blockIdx.x;
    unsigned int bdx 		= blockDim.x;
    unsigned int gdx 		= gridDim.x;
    unsigned int i 			= bix*bdx + tid;
    unsigned int gridSize 	= bdx*gdx;

    while (i < map_len)
    {
    	//BIN[i] *= ROI[i];
    	BIN[i] = (unsigned char) ((int)BIN[i] * (int)ROI[i]);
        i += gridSize;
    }
}

__global__ void imperviousness_change(
							const unsigned char *dev_BIN1, const unsigned char *dev_BIN2,
							unsigned int WIDTH, unsigned int HEIGHT, int *dev_LTAKE_map
							)
{
	unsigned long int x 	= threadIdx.x;
	unsigned long int bdx	= blockDim.x;
	unsigned long int bix	= blockIdx.x;
	unsigned long int tix	= bdx*bix + x;	// offset

	if( tix < WIDTH*HEIGHT ){
		dev_LTAKE_map[tix]	= (int)((int)dev_BIN2[tix] - (int)dev_BIN1[tix]);
	}
}

__global__ void imperviousness_change_double(
							const unsigned char *dev_BIN1, const unsigned char *dev_BIN2,
							unsigned int WIDTH, unsigned int HEIGHT, double *dev_LTAKE_map
							)
{
	unsigned long int x 	= threadIdx.x;
	unsigned long int bdx	= blockDim.x;
	unsigned long int bix	= blockIdx.x;
	unsigned long int tix	= bdx*bix + x;	// offset

	if( tix < WIDTH*HEIGHT ){
		dev_LTAKE_map[tix]	= (double) ( (double)dev_BIN2[tix] - (double)dev_BIN1[tix] );
	}
}
__global__ void imperviousness_change_char(
							const unsigned char *dev_BIN1, const unsigned char *dev_BIN2,
							unsigned int WIDTH, unsigned int HEIGHT, char *dev_LTAKE_map
							)
{
	unsigned long int x 	= threadIdx.x;
	unsigned long int bdx	= blockDim.x;
	unsigned long int bix	= blockIdx.x;
	unsigned long int tix	= bdx*bix + x;	// offset

	if( tix < WIDTH*HEIGHT ){
		dev_LTAKE_map[tix]	= dev_BIN2[tix] - dev_BIN1[tix];
	}
}

__global__ void imperviousness_change_large(
							const unsigned char *dev_BIN1, const unsigned char *dev_BIN2,
							unsigned int WIDTH, unsigned int HEIGHT, int *dev_LTAKE_map,
							int mapel_per_thread
							)
{
	unsigned long int x 	= threadIdx.x;
	unsigned long int bdx	= blockDim.x;
	unsigned long int bix	= blockIdx.x;
	//unsigned long int gdx	= gridDim.x;
	unsigned long int tid	= bdx*bix + x;	// offset
	unsigned long int tix	= tid * mapel_per_thread;	// offset

	//extern __shared__ int sh_diff[];

	if( bdx*bix*mapel_per_thread < WIDTH*HEIGHT ){
		//sh_diff[tid] = 0; syncthreads();
		for(long int ii=0;ii<mapel_per_thread;ii++){
			if( tix+ii < WIDTH*HEIGHT ){
				//sh_diff[tid]		= (int)((int)dev_BIN2[tix+ii] - (int)dev_BIN1[tix+ii]);
				dev_LTAKE_map[tix+ii] = (int)((int)dev_BIN2[tix+ii] - (int)dev_BIN1[tix+ii]);
			} //__syncthreads();
			//dev_LTAKE_map[tix+ii]	= sh_diff[tid];
		}
	}
}

__global__ void imperviousness_change_histc_sh_4(
							const unsigned char *dev_BIN1, const unsigned char *dev_BIN2,
							unsigned int WIDTH, unsigned int HEIGHT,
							int *dev_LTAKE_count, int mapel_per_thread
							)
{
/*
 INPUTS
    dev_BIN1:			Imperviousness of OLDER year.
    dev_BIN1:			Imperviousness of NEWER year.

 OUTPUTS
    dev_LTAKE_count:	2x2 table with counts about the 4 possible combinations.
    dev_LTAKE_map:      map storing the difference (dev_BIN2-dev_BIN1).
    					The following 4 combinations are possible computing the difference:
    						---------------------------
    						(N) (BIN2,BIN1)	--> (LTAKE)
    						---------------------------
    						(1) (0,0) 		--> +0			---> nothing changed in rural pixels
    						(2) (0,1) 		--> -1			---> increase of rural pixels
    						(3) (1,0) 		--> +1			---> increase of urban pixels
    						(4) (1,1) 		--> -0			---> nothing changed in urban pixels
    						---------------------------
    					where values can be { 0:rural; 1:urban }.

 DESCRIPTION
 	 This kernel function counts the number of pixels for each LTAKE type (i.e. {+0,-1,+1,-0}).
 	 It assumes that:
 	 	 > the number of outcomes LTAKE can assume is equal to FOUR=4, as also stated in kernel name "..._4"
 	 	 > each thread within a block is in charge of mapel_per_thread pixels in order to allocate
 	 	   a number of blocks equal to the number of available SMs.
 	 	 > the number of threads per block is equal to 256(=bdx).
 	 I have to call this kernel using the following resources:
 	 	 > block:	(bdx*mapel_per_thread, 1, 1)
 	 	 > sh_mem:	bdx*4*sizeof(int)
*/

	unsigned int x 			= threadIdx.x;
	unsigned int bdx		= blockDim.x;
	unsigned int bix		= blockIdx.x;
	unsigned int tid		= (bdx*bix + x);		// global thread index
	unsigned int tix		= tid*mapel_per_thread;	// offset, considering mapel_per_thread pixels per thread

	const int num_bins 		= 4;
	const int nclasses		= num_bins/2;

	extern __shared__ int sh_sum[];
	int loc_sum[num_bins];
	unsigned int ii, o;

//	if( tix < (WIDTH*HEIGHT) ){ // - WIDTH*HEIGHT%mapel_per_thread+1
		// initialise at zero (IMPORTANT!!!):
		for(ii=0;ii<num_bins;ii++){
			loc_sum[ii]				= 0;
			sh_sum[x*num_bins+ii]	= 0;
		}
		syncthreads();

		// compute difference and store the count in local memory
		// (each thread is in charge of mapel_per_thread map elements):
		for(ii=0;ii<mapel_per_thread;ii++){
			if(tix+ii<WIDTH*HEIGHT) loc_sum[dev_BIN2[tix+ii]*nclasses+dev_BIN1[tix+ii]] += 1;
		}

		// copy from local to shared memory:
		for(ii=0;ii<num_bins;ii++) sh_sum[ii*bdx+x]		= loc_sum[ii];
		syncthreads();

		// reduce two bins per time (to maximise warp allocation
		for(ii=0;ii<num_bins;ii++){
			o = ii*bdx;
			if(x<128)		sh_sum[x+o] += sh_sum[x+o + 128];	syncthreads();
			if(x<64) 		sh_sum[x+o] += sh_sum[x+o + 64];	syncthreads();
			if(x<32) 		sh_sum[x+o] += sh_sum[x+o + 32];	syncthreads();
			if(x<16) 		sh_sum[x+o] += sh_sum[x+o + 16];	syncthreads();
			if(x<8) 		sh_sum[x+o] += sh_sum[x+o + 8];		syncthreads();
			if(x<4) 		sh_sum[x+o] += sh_sum[x+o + 4];		syncthreads();
			if(x<2) 		sh_sum[x+o] += sh_sum[x+o + 2];		syncthreads();
			if(x<1) 		sh_sum[x+o] += sh_sum[x+o + 1];		syncthreads();
			// each bix writes his count value:
			if(x==0)		atomicAdd( &dev_LTAKE_count[ii], sh_sum[x+o] );
/*			if(x>=bdx/2){
				o = ii*bdx*2 + bdx/2;
								sh_sum[x+o] += sh_sum[x+o + 128];	syncthreads();
				if(x<bdx/2+64)	sh_sum[x+o] += sh_sum[x+o + 64];	syncthreads();
				if(x<bdx/2+32)	sh_sum[x+o] += sh_sum[x+o + 32];	syncthreads();
				if(x<bdx/2+16)	sh_sum[x+o] += sh_sum[x+o + 16];	syncthreads();
				if(x<bdx/2+8)	sh_sum[x+o] += sh_sum[x+o + 8];		syncthreads();
				if(x<bdx/2+4)	sh_sum[x+o] += sh_sum[x+o + 4];		syncthreads();
				if(x<bdx/2+2)	sh_sum[x+o] += sh_sum[x+o + 2];		syncthreads();
				if(x<bdx/2+1)	sh_sum[x+o] += sh_sum[x+o + 1];		syncthreads();
				// each bix writes his count value:
				if(x<bdx/2+1)	atomicAdd( &dev_LTAKE_count[1+ii*2], sh_sum[x+o] );
			}
*/		}
	//}
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

	metadata 			MDbin,MDroi,MDdouble;
	unsigned int		map_len;
	double 				*dev_LTAKE_map,*host_LTAKE_map;
	//int 				*dev_LTAKE_map,*host_LTAKE_map;
	//char 				*dev_LTAKE_map,*host_LTAKE_map;
	int 				*dev_LTAKE_count,*host_LTAKE_count;
	//double 				*dev_LTAKE_count,*host_LTAKE_count;
	unsigned char		*dev_BIN1, *dev_BIN2, *dev_ROI;
	clock_t				start_t,end_t;
	unsigned int 		elapsed_time		= 0;
	cudaDeviceProp		devProp;
	unsigned int		gpuDev=0;
	// count the number of kernels that must print their output:
	unsigned int 		count_print 		= 0;
	bool 				use_large			= false;
	int 				mapel_per_thread_2 	= 0;

	/*
	 * 		LOAD METADATA & DATA
	 */
	MDbin					= geotiffinfo( FIL_BIN1, 1 );
	MDroi 					= geotiffinfo( FIL_ROI,  1 );
	// set metadata to eventually print arrays after any CUDA kernel:
/*	MDint 					= MDbin;
	MDint.pixel_type		= GDT_Int32;
*/	MDdouble 				= MDbin;
	MDdouble.pixel_type		= GDT_Float64;
	// Set size of all arrays which come into play:
	map_len 				= MDbin.width*MDbin.heigth;
	size_t	sizeChar		= map_len*sizeof( unsigned char );
	size_t	sizeInt			= 4*sizeof( int 			);
	size_t	sizeDouble		= map_len*sizeof( double 		);
	// initialize arrays:
	unsigned char *BIN1		= (unsigned char *) CPLMalloc( sizeChar );
	unsigned char *BIN2		= (unsigned char *) CPLMalloc( sizeChar );
	unsigned char *ROI 		= (unsigned char *) CPLMalloc( sizeChar );
	// load ROI:
	printf("Importing...\t%s\n",FIL_ROI);
	geotiffread( FIL_ROI, MDroi, &ROI[0]   );
	// load BIN:
	printf("Importing...\t%s\n",FIL_BIN1);
	geotiffread( FIL_BIN1, MDbin, &BIN1[0] );
	printf("Importing...\t%s\n",FIL_BIN2);
	geotiffread( FIL_BIN2, MDbin, &BIN2[0] );

	/*
	 * 	INITIALIZE CPU & GPU ARRAYS
	 */
	// initialize grids on CPU MEM:
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_LTAKE_map, 	sizeDouble)  );
	//CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_LTAKE_map, 	sizeInt)  );
	//CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_LTAKE_map, 	sizeChar)  );
	//CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_LTAKE_count, 	4*sizeof(double))  );
	CUDA_CHECK_RETURN( cudaMallocHost( 	(void**)&host_LTAKE_count, 	sizeInt)  );
	// initialize grids on GPU MEM:
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_BIN1, 		sizeChar) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_BIN2, 		sizeChar) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_ROI,  		sizeChar) );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_LTAKE_map, 	sizeDouble)  );
	//CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_LTAKE_map, 	sizeInt)  );
	//CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_LTAKE_map, 	sizeChar)  );
	//CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_LTAKE_count, 	4*sizeof(double))  );
	CUDA_CHECK_RETURN( cudaMalloc(		(void **)&dev_LTAKE_count, 	sizeInt)  );
	// memset:
	CUDA_CHECK_RETURN( cudaMemset(dev_LTAKE_count, 0, sizeInt) );
	//CUDA_CHECK_RETURN( cudaMemset(dev_LTAKE_count, 0, 4*sizeof(double)) );
	// H2D:
	CUDA_CHECK_RETURN( cudaMemcpy(dev_BIN1, BIN1, 	sizeChar, cudaMemcpyHostToDevice) );
	CUDA_CHECK_RETURN( cudaMemcpy(dev_BIN2, BIN2, 	sizeChar, cudaMemcpyHostToDevice) );
	CUDA_CHECK_RETURN( cudaMemcpy(dev_ROI,  ROI, 	sizeChar, cudaMemcpyHostToDevice) );

	/*
	 * 		QUERY CURRENT GPU PROPERTIES
	 */
	CUDA_CHECK_RETURN( cudaSetDevice(gpuDev) );
	cudaGetDeviceProperties(&devProp, gpuDev);
	int N_sm				= devProp.multiProcessorCount;
	int max_threads_per_SM	= devProp.maxThreadsPerMultiProcessor;
//	int max_shmem_per_block	= devProp.sharedMemPerBlock;

	/*
	 * 		KERNELS GEOMETRY
	 * 		NOTE: use ceil() instead of the "%" operator!!!
	 */
	unsigned int bdx, gdx, num_blocks_per_SM, mapel_per_thread, Nblks_per_grid;
/*	if(map_len/BLOCK_DIM < N_sm){
		bdx					= BLOCK_DIM_small;
	}else {
		bdx					= BLOCK_DIM;
	}
*/
	bdx						= BLOCK_DIM;
	num_blocks_per_SM       = max_threads_per_SM / bdx;
	mapel_per_thread        = (unsigned int)ceil( (double)map_len / (double)((bdx*1)*N_sm*num_blocks_per_SM) );
	gdx                     = (unsigned int)ceil( (double)map_len / (double)( mapel_per_thread*(bdx*1) ) );
	Nblks_per_grid 			= N_sm* (max_threads_per_SM /threads);

	dim3 block( bdx,1,1 );
	dim3 grid ( gdx,1,1 );
	dim3 	dimBlock( threads, 1, 1 );
	dim3 	dimGrid(  Nblks_per_grid,  1, 1 );
	int sh_mem				= (bdx*4)*(sizeof(int));
	//double sh_mem_double	= (bdx*4)*(sizeof(double));
	unsigned int gdx_2		= (unsigned int)ceil( (double)map_len / (double)( (bdx*4) ) );
	dim3 block_2( bdx*4,1,1 );
	dim3 grid_2 ( gdx_2,1,1 );
	if (gdx_2>devProp.maxGridSize[0]) {
/*		printf( "Error: cannot allocate gridsize=%d (limit is %d)!\n",gdx_2,devProp.maxGridSize[0] );
		exit(EXIT_FAILURE);
*/		use_large 			= true;
		mapel_per_thread_2 	= 32*4; // warp size times 4
		gdx_2				= (unsigned int)ceil( (double)map_len / (double)( (bdx*2*mapel_per_thread_2) ) );
		dim3 block_2( bdx*2,1,1 );
		dim3 grid_2 ( gdx_2,1,1 );
	}

	/*		KERNELS INVOCATION
	 *
	 *			********************************
	 *			-0- filter_roi twice(BIN1,BIN2)
	 *			-1- imperviousness_change_sh_4
	 *			-2- imperviousness_change
	 *			********************************
	 *
	 *		Note that imperviousness_change_large does not work!!
	 */
	printf("\n\n");
	// ***-0-***
	start_t = clock();
	filter_roi<<<dimGrid,dimBlock>>>(dev_BIN1,dev_ROI,map_len);
	filter_roi<<<dimGrid,dimBlock>>>(dev_BIN2,dev_ROI,map_len);
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %34s\t%6d [msec]\n",++count_print,kern_0,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(host_LTAKE_count,dev_LTAKE_count,	(size_t)4*sizeof(int),cudaMemcpyDeviceToHost) );
		//CUDA_CHECK_RETURN( cudaMemcpy(host_LTAKE_count,dev_LTAKE_count,	(size_t)4*sizeof(double),cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.txt",BASE_PATH,count_print,kern_0);
		write_mat_int( host_LTAKE_count, 4, 1, buffer );
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

	// ***-1-***
	start_t = clock();
	imperviousness_change_histc_sh_4<<<grid,block,sh_mem>>>( 	dev_BIN1, dev_BIN2, MDbin.width, MDbin.heigth,
																dev_LTAKE_count, mapel_per_thread );

/*	imperviousness_change_histc_sh_4_double<<<grid,block,sh_mem_double>>>(
														dev_BIN1, dev_BIN2, MDbin.width, MDbin.heigth,
														dev_LTAKE_count, mapel_per_thread );
*/	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %34s\t%6d [msec]\n",++count_print,kern_1,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(host_LTAKE_count,dev_LTAKE_count,	(size_t)sizeInt,cudaMemcpyDeviceToHost) );
		//CUDA_CHECK_RETURN( cudaMemcpy(host_LTAKE_count,dev_LTAKE_count,	(size_t)4*sizeof(double),cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.txt",BASE_PATH,count_print,kern_1);
		write_mat_int( host_LTAKE_count, 4, 1, buffer );
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:
	// ***-2-***
	start_t = clock();
	if(use_large!=true){
		imperviousness_change_double<<<grid_2,block_2>>>( 		dev_BIN1, dev_BIN2, MDbin.width, MDbin.heigth, dev_LTAKE_map );
		//imperviousness_change<<<grid_2,block_2>>>( 		dev_BIN1, dev_BIN2, MDbin.width, MDbin.heigth, dev_LTAKE_map );
		//imperviousness_change_char<<<grid_2,block_2>>>( dev_BIN1, dev_BIN2, MDbin.width, MDbin.heigth, dev_LTAKE_map );
	}else{
		printf("Error: imperviousness_change_large does not work yet!");
		exit(EXIT_FAILURE);
/*		imperviousness_change_large<<<grid_2,block_2>>>(dev_BIN1, dev_BIN2, MDbin.width, MDbin.heigth, dev_LTAKE_map, mapel_per_thread_2 );
		kern_2 		= "imperviousness_change_large"	;
*/	}
	CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	end_t = clock();
	printf("  -%d- %34s\t%6d [msec]\n",++count_print,kern_2,(int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 ));
	if (print_intermediate_arrays){
		CUDA_CHECK_RETURN( cudaMemcpy(host_LTAKE_map,dev_LTAKE_map,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
		//CUDA_CHECK_RETURN( cudaMemcpy(host_LTAKE_map,dev_LTAKE_map,	(size_t)sizeInt,cudaMemcpyDeviceToHost) );
		//CUDA_CHECK_RETURN( cudaMemcpy(host_LTAKE_map,dev_LTAKE_map,	(size_t)sizeChar,cudaMemcpyDeviceToHost) );
		sprintf(buffer,"%s/data/-%d-%s.tif",BASE_PATH,count_print,kern_2);
		geotiffwrite( FIL_BIN1, buffer, MDdouble, host_LTAKE_map );
		//geotiffwrite( FIL_BIN1, buffer, MDint, host_LTAKE_map );
		//geotiffwrite( FIL_BIN1, buffer, MDbin, host_LTAKE_map );
	}
	elapsed_time += (int)( (double)(end_t  - start_t ) / (double)CLOCKS_PER_SEC * 1000 );// elapsed time [ms]:

	printf("________________________________________________________________\n");
	printf("%40s\t%6d [msec]\n", "Total time:",elapsed_time );

	CUDA_CHECK_RETURN( cudaMemcpy(host_LTAKE_map,dev_LTAKE_map,	(size_t)sizeDouble,cudaMemcpyDeviceToHost) );
	//CUDA_CHECK_RETURN( cudaMemcpy(host_LTAKE_map,dev_LTAKE_map,	(size_t)sizeInt,cudaMemcpyDeviceToHost) );
	//CUDA_CHECK_RETURN( cudaMemcpy(host_LTAKE_map,dev_LTAKE_map,	(size_t)sizeChar,cudaMemcpyDeviceToHost) );
	CUDA_CHECK_RETURN( cudaMemcpy(host_LTAKE_count,dev_LTAKE_count,	(size_t)sizeInt,cudaMemcpyDeviceToHost) );
	//CUDA_CHECK_RETURN( cudaMemcpy(host_LTAKE_count,dev_LTAKE_count,	(size_t)4*sizeof(double),cudaMemcpyDeviceToHost) );
	// save on HDD
	geotiffwrite( FIL_BIN1, FIL_LTAKE_grid, MDdouble, host_LTAKE_map );
	//geotiffwrite( FIL_BIN1, FIL_LTAKE_grid, MDint, host_LTAKE_map );
	//geotiffwrite( FIL_BIN1, FIL_LTAKE_grid, MDbin, host_LTAKE_map );

	write_mat_T( host_LTAKE_count, 4, 1, FIL_LTAKE_count );

	// CUDA free:
	cudaFree( dev_BIN1			);
	cudaFree( dev_BIN2			);
	cudaFree( dev_LTAKE_map		);
	cudaFree( dev_LTAKE_count	);
	cudaFree( dev_ROI			);

	cudaFree( BIN1				);
	cudaFree( BIN2				);
	cudaFree( host_LTAKE_map	);
	cudaFree( host_LTAKE_count	);
	cudaFree( ROI				);

	// Destroy context
	CUDA_CHECK_RETURN( cudaDeviceReset() );

	printf("\n\n\nFinished!!\n");

	return 0;// elapsed_time
}
