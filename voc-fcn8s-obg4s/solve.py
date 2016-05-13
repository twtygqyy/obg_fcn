import caffe,sys
sys.path.append("../")
import surgery, score

import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

weights = 'voc-fcn8s-obg4s.caffemodel'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

interp_layers = [k for k in solver.net.params.keys() if 'mask_projection' in k]
surgery.initial(solver.net, interp_layers)

# scoring
val = np.loadtxt('aug_val.txt', dtype=str)

for _ in range(50):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')
    

