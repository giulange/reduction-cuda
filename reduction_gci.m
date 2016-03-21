%% PARS
BASE_DIR                = '/home/giuliano/git/cuda/reduction';
% I/–
% FIL_BIN1                = fullfile( BASE_DIR, 'data', 'BIN1.tif');
% FIL_BIN2                = fullfile( BASE_DIR, 'data', 'BIN2.tif');
% FIL_ROI                 = fullfile( BASE_DIR, 'data', 'ROI.tif');
FIL_ROI                 = '/media/DATI/db-backup/ssgci-data/testing/ssgci_roi.tif';
FIL_BIN1                = '/media/DATI/db-backup/ssgci-data/testing/ssgci_bin.tif';
FIL_BIN2                = '/media/DATI/db-backup/ssgci-data/testing/ssgci_bin2.tif';
% –/O
FIL_LTAKE_grid          = fullfile(BASE_DIR,'data','LTAKE_map.tif');
FIL_LTAKE_count         = fullfile(BASE_DIR,'data','LTAKE_count.txt');

% cuda exe:
exefil                  = fullfile( BASE_DIR, 'Debug', 'reduction-gci' );

% pars:
create_bins             = 0;
cu_run                  = 1;
plot_me                 = 0;
%% create two BINs and save (once)
if create_bins
    % parameters:
    threshold               = 0.6;
    newsize                 = [8000,8000];% 
    % NOTE: if I use [9000,9000] the imperviousness_change_large kernel should
    % be called, but this kernel does not work fine yet!!
    % The limit is given by:
    cGPU = gpuDevice(1);
    Threshold = cGPU.MaxThreadsPerBlock * cGPU.MaxGridSize(1);

    if prod(newsize)>Threshold
        error(  'Exceded the maximum number of pixels (%d) that the basic %s kernel can handle!',...
                Threshold,'"imperviousness_change"')
    end

    % create:
    ROI                     = true( newsize );
    BIN1                    = rand( newsize );
    BIN2                    = rand( newsize );
    BIN1(BIN1>=threshold)   = 1;
    BIN1(BIN1< threshold)   = 0;
    BIN2(BIN2>=threshold)   = 1;
    BIN2(BIN2< threshold)   = 0;
    BIN1                    = logical(BIN1);
    BIN2                    = logical(BIN2);

    % Build the new georeferenced GRID:
    info                    = geotiffinfo( fullfile('/home/giuliano/git/cuda/fragmentation/data','BIN.tif') );
    R                       = info.SpatialRef;
    newXlim                 = [ R.XLimWorld(1), R.XLimWorld(1) + newsize(2)*R.DeltaX ];
    newYlim                 = [ R.YLimWorld(1), R.YLimWorld(1) + newsize(1)*(-R.DeltaY) ];
    Rnew                    = maprasterref( ...
                'XLimWorld',              newXlim, ...
                'YLimWorld',              newYlim, ...
                'RasterSize',             newsize, ...
                'RasterInterpretation',   R.RasterInterpretation, ...
                'ColumnsStartFrom',       R.ColumnsStartFrom, ...
                'RowsStartFrom',          R.RowsStartFrom ...
                                      );

    geotiffwrite( fullfile(BASE_DIR,'data',FIL_ROI),   ROI, Rnew, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag );
    geotiffwrite( fullfile(BASE_DIR,'data',FIL_BIN1), BIN1, Rnew, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag );
    geotiffwrite( fullfile(BASE_DIR,'data',FIL_BIN2), BIN2, Rnew, 'GeoKeyDirectoryTag', info.GeoTIFFTags.GeoKeyDirectoryTag );
end
%% import BINs & ROI
BIN1                    = geotiffread( FIL_BIN1 );
BIN2                    = geotiffread( FIL_BIN2 );
ROI                     = geotiffread( FIL_ROI  );

if plot_me
    imshowpair(BIN1,BIN2,'montage') %, 'InitialMagnification','fit')
end
%% compute MatLab Land Take
tic
BIN1f       = BIN1;
BIN1f(ROI==0) = 0;
BIN2f       = BIN2;
BIN2f(ROI==0) = 0;
% kernel 1
count(1)    = sum(BIN2f(:)==0 & BIN1f(:)==0);
count(2)    = sum(BIN2f(:)==0 & BIN1f(:)==1);
count(3)    = sum(BIN2f(:)==1 & BIN1f(:)==0);
count(4)    = sum(BIN2f(:)==1 & BIN1f(:)==1);
% kernel 2
LTAKE_ml    = double(BIN2f) - double(BIN1f);
T(1)        = toc;
%% execute CUDA-code & grab stout
if cu_run
    
    setenv('LD_LIBRARY_PATH', '');
    
    % run_pars        = [' ',num2str(tiledimX),' ',num2str(tiledimY),' ',num2str(WIDTH),' ',num2str(HEIGHT),' ',num2str(print_me)];
    run_pars        = '';
    run_str         = [exefil, run_pars];
    fprintf('\n%s\n',repmat('–',1,130))
    fprintf('Running CUDA-C program:\n\t%s\n',run_str);
    fprintf('%s\n',repmat('–',1,130))
    tic
    [status,out]    = unix( run_str, '-echo' );
    T(3)            = toc;
    fprintf('%s\n\n\n',repmat('–',1,130))
    sq_open         = strfind(out,'[');
    str_divide      = strfind(out,':');
    T(2)            = str2double( out(str_divide(end)+1:sq_open(end) -1) )/1000;
end
%% print bins & counts [matlab vs cuda]
% -----------------------------
% (N) (BIN2,BIN1)	--> (LTAKE)
% -----------------------------
% (1) (0,0) 		--> +0			---> nothing changed in rural pixels
% (2) (0,1) 		--> -1			---> increase of rural pixels
% (3) (1,0) 		--> +1			---> increase of urban pixels
% (4) (1,1) 		--> -0			---> nothing changed in urban pixels
% -----------------------------
% where values can be { 0:rural; 1:urban }.

%--OLD--
% bins    = [-1,0,+1];
% count   = histc(LTAKE_ml(:),bins);
%--OLD--

bins = {'+0','-1','+1','-0'};

% MATLAB
fprintf('\n')
fprintf('%s\n','MatLab')
fprintf('%s\n',repmat('-',1,16))
fprintf(' %s\t%s\n','bins','hist')
fprintf('%s\n',repmat('-',1,16))
for ii = 1:size(bins,2)
    fprintf(' %s\t%d\n',bins{ii},count(ii))
end
fprintf('%s\n',repmat('-',1,16))
fprintf(' %s\t%d\n','tot',sum(count))

fprintf('\n')

% CUDA
count_cu = load( FIL_LTAKE_count );
fprintf('\n')
fprintf('%s\n','CUDA')
fprintf('%s\n',repmat('-',1,16))
fprintf(' %s\t%s\n','bins','hist')
fprintf('%s\n',repmat('-',1,16))
for ii = 1:size(bins,2)
    good = 'x';
    fprintf(' %s\t%d',bins{ii},count_cu(ii))
    if count_cu(ii)-count(ii)==0, good = 'ok'; end
    cFormat = ['%',num2str(14-numel(num2str(count_cu(ii)))),'s!'];
    fprintf(cFormat,good)
    fprintf('\n')
end
fprintf('%s\n',repmat('-',1,16))
good = 'x';
fprintf(' %s\t%d','tot',sum(count_cu))
if sum(count_cu)-sum(count)==0, good = 'ok'; end
cFormat = ['%',num2str(14-numel(num2str(sum(count_cu)))),'s!'];
fprintf(cFormat,good)
fprintf('\n')

fprintf('\n')
%% import Cuda-C Land Take
LTAKE_cu        = geotiffread( FIL_LTAKE_grid );
%% DIFF MatLab vs CUDA
DIFF            = LTAKE_ml-LTAKE_cu;
fprintf( '  > No. pixels with errors (ml-cu):\t%d\n', sum(abs( DIFF(:) )) )
fprintf( '  > speed-up (ml/cu) = %3.1f\t( %3.2f [ms] / %3.2f [ms] )\n', T(1)/T(2),T(1)*1000,T(2)*1000 );
%% deep check
[r,c]=find(DIFF);
if ~isempty( r )
    fprintf('%12s, %12s, %12s, %12s, %12s, %12s\n', ...
            'BIN1','BIN2','BIN1f','BIN2f','LTAKE[cu]','LTAKE[ml]' ...
           )
    for ii = 1:10
        fprintf('%12d, %12d, %12d, %12d, %12d, %12d\n',     ...
            BIN1(r(ii),c(ii)), BIN2(r(ii),c(ii)),           ...
            BIN1f(r(ii),c(ii)), BIN2f(r(ii),c(ii)),         ...
            LTAKE_cu(r(ii),c(ii)), LTAKE_ml(r(ii),c(ii))    ...
        )
    end
end
%% understand/check CUDA grid
if ~create_bins, newsize = size(BIN1); end
maplen  = prod(newsize);
mapel_per_thread = 32*4;
bdx     = 512;
gdx     = ceil( maplen / (mapel_per_thread*bdx) );

% x       = 0:bdx-1;
% bix     = 0:gdx-1;

% extremes:
x       = [0,bdx-1];
bix     = [0,gdx-1];

tid     = bdx*bix+x;
tix     = tid*mapel_per_thread;
%% understand grid/block size for imperviousness_change_sh4
% % set current GPU:
% cGPU                    = gpuDevice(2);
% % number of Streaming Multiprocessors:
% N_sm                    = cGPU.MultiprocessorCount;
% 
% % fetch metadata:
% info                    = geotiffinfo( fullfile('/home/giuliano/git/cuda/reduction/data','BIN1.tif') );
% % fetch image size from file:
% map_len                 = info.Height*info.Width;
% 
% % number of threads per block (on X dim, since Y=1):
% max_threads_per_SM      = 1536; % maxThreadsPerMultiProcessor ==> deviceQuery.cpp
% bdx                     = 16*16;
% bdy                     = 1;
% 
% % number of map elements in charge of each thread:
% % I assume that:
% %   > one map element per thread is time consuming;
% %   > the number of active blocks equals the number of streaming
% %     multiprocessors (instead I should consider the maximum number of
% %     blocks per SM and the maximum number of threads per SM in order to
% %     correctly calculate the number of map elements per thread). I raise
% %     the number of map elements per thread and reduce the overall number
% %     of atomic operations (atomicAdd) performed (expecting to be faster);
% %   > I can decide to assign 16 map elements to each thread and calculate
% %     the CUDA grid dimension in X accordingly. This way I raise the number
% %     of total blocks per GPU (assuming I can allocate more blocks per SM)
% %     but I raise the number of atomic operations (atomicAdd) performed
% %     (hence I expect to be slower). But how can I know the number of
% %     blocks per SM I have on board??
% % 
% % number of blocks per SM to reach full occupancy
% num_blocks_per_SM       = max_threads_per_SM / bdx;
% mapel_per_thread        = ceil( map_len / ((bdx*bdy)*N_sm*num_blocks_per_SM) );
% gdx                     = ceil( map_len / ( mapel_per_thread*(bdx*bdy) ) );
% gdy                     = 1;
% 
% x                       = 0:bdx-1; % any thread value in range [1,bdx]
% bix                     = gdx-1; % or another value in range [1,gdx]
% tix                     = (bix*bdx+x)*mapel_per_thread;
