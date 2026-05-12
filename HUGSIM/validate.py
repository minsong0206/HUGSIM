import torch; print('torch:', torch.__version__)
from simple_knn._C import distCUDA2; print('simple_knn: OK')
from unidepth.models import UniDepthV2; print('unidepth: OK')
import numpy as np; print('numpy:', np.__version__)
try:
    import xformers; print('xformers:', xformers.__version__)
except:
    print('xformers: not installed')
print('=== ALL PASSED ===')
