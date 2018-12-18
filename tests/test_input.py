


import openConv as oc
import numpy as np
from nose.tools import *


@raises(NotImplementedError)
def test_methodNotImplemented():
    oc.Conv(10, 2, None, np.zeros(29), 2, 1., 20, method = -1)

