
import grid2op
from newtonBackend import NewtonBackend

# or any other name.  -> this is sent to the backend `/home/whatehver/grid2op_datasets/l2rpn_case14_sandbox/grid.json`
env_name = "l2rpn_case14_sandbox"
env = grid2op.make(env_name, backend=NewtonBackend())
