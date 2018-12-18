


import openConv as oc
from nose.tools import *


@raises(NotImplementedError)
def test_methodNotImplemented():
    oc.Conv(10, 2, None, np.zeros(29), 2, 1., 20, method = -1)


#@nottest
#def helper_test_method(nData, forwardBackward, shift, methods, orders, rtol):

#    return None
