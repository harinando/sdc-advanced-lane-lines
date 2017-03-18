BATCH_SIZE = 128
EPOCHS = 30
FORMAT = '%(asctime)-15s %(clientip)s %(user)-8s %(message)s'
# WIDTH = 128
# HEIGHT = 64
# WIDTH = 320
# HEIGHT = 160
WIDTH = 160
HEIGHT = 80
DEPTH = 3
ADJUSTMENT = 0.005
ALPHA = 0.001
DROPOUT = 0.5
OUTPUT = '.hdf5_checkpoints'
COLUMNS = ["center", "left", "right", "steering", "throttle", "brake", "speed", "position","rotation"]
SKIP = ["dataset_sim_000_km_few_laps"]
SCALER = 'scaler.p'
COLUMNS_TO_NORMALIZE = ['positionX', 'positionY', 'positionZ', 'rotationX', 'rotationY', 'rotationZ']
FEATURES = ['positionX', 'positionY', 'positionZ', 'rotationX', 'rotationY', 'rotationZ', 'speed']


SVR_MODEL = 'svr.p'
SCALER = 'scaler.p'