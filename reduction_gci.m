%% PARS
BASE_DIR                = '/home/giuliano/git/cuda/reduction';
FIL_BIN1                = 'BIN1.tif';
FIL_BIN2                = 'BIN2.tif';
FIL_ROI                 = 'ROI.tif';
%% create two BINs and save (once)

% parameters:
threshold               = 0.5;
newsize                 = [256,256];

% create:
ROI                     = ones( newsize );
BIN1                    = rand( newsize );
BIN2                    = rand( newsize );
BIN1(BIN1>=threshold)   = 1;
BIN1(BIN1< threshold)   = 0;
BIN2(BIN2>=threshold)   = 1;
BIN2(BIN2< threshold)   = 0;

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
%% import BINs & ROI
BIN1                    = geotiffread( fullfile(BASE_DIR,'data',FIL_BIN1) );
BIN2                    = geotiffread( fullfile(BASE_DIR,'data',FIL_BIN2) );
ROI                     = geotiffread( fullfile(BASE_DIR,'data',FIL_ROI) );
%% compute

DIFF    = (BIN2-BIN1).*ROI;
SUM     = (BIN2+BIN1).*ROI;
bins    = [-1,0,+1];
count   = histc(DIFF(:),bins);

fprintf('\n')
fprintf('%s\n',repmat('-',1,16))
fprintf(' %s\t%s\n','bins','hist')
fprintf('%s\n',repmat('-',1,16))
for ii = 1:length(bins)
    fprintf(' %+2d\t%d\n',bins(ii),count(ii))
end
fprintf('%s\n',repmat('-',1,16))
fprintf('\n')
