
from snntoolbox.simulation.target_simulators.pyNN_target_sim import SNN
from snntoolbox.bin.utils import update_setup

#import pyNN.connectors
#import pyNN.nest as nest

import numpy as np


#
# l_0 = pyNN.connectors.FromFileConnector("test-dir/agentzoo-conversion/00Dense_128")
# b_0 = pyNN.connectors.FromFileConnector("test-dir/agentzoo-conversion/00Dense_128_biases")
#
# l_1 = pyNN.connectors.FromFileConnector("test-dir/agentzoo-conversion/01Dense_64")
# b_1 = pyNN.connectors.FromFileConnector("test-dir/agentzoo-conversion/01Dense_64_biases")
#
# l_2 = pyNN.connectors.FromFileConnector("test-dir/agentzoo-conversion/02Dense_8")
# b_2 = pyNN.connectors.FromFileConnector("test-dir/agentzoo-conversion/02Dense_8_biases")
#


config = update_setup("config-agentzoo.ini")
snn = SNN(config)
snn.load("test-dir/agentzoo-conversion/", "keras-ant2_nest")

if not snn.is_built:
    snn.restore_snn()

snn.init_cells()

ob = np.random.randn(28)

test_data = {
    "x_b_l": ob
}

output = snn.simulate(**test_data)
#snn.run()

print(output)


#assembly = nest.Assembly(*layers)


##assembly.inject()