OUTPUT = 'output_images'
TEST = 'test_images'


CARS_FEAT = '{}/cars.p'.format(OUTPUT)
NOTCARS_FEAT = '{}/notcars.p'.format(OUTPUT)
SVC_MODEL = '{}/SVC.p'.format(OUTPUT)

SCALER = '{}/SCALER.p'.format(OUTPUT)

OVERLAP = 0.5

COLOR_SPACE = 'YCrCb'
ORIENT = 9
PIX_PER_CELL = 8
CELL_PER_BLOCK = 2
HOG_CHANNEL = 'ALL' # 0, 1, 2, 'ALL'
SPATIAL_SIZE = (32, 32)
HIST_BIN = 32
SPATIAL_FEAT = True # Spatial features on or off
HIST_FEAT = True    # Histogram features on or off
HOG_FEAT = True     # Hog features on or off
YSTART = 400
YSTOP = 656
SCALE = 1.5