import caffe,sys
sys.path.append("../")
import surgery, score

import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = 'train-no-bais_iter_416000.caffemodel'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
#solver.net.copy_from(weights)
solver.restore('train-no-bais_iter_20000.solverstate')

# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'upscore8' in k]
#surgery.interp(solver.net, interp_layers)

# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'mask_projection' in k]
#surgery.initial(solver.net, interp_layers)


# scoring
val = np.loadtxt('aug_val.txt', dtype=str)

for _ in range(30):
    score.seg_tests(solver, False, val, layer='score')
    solver.step(4000)

