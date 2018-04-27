import numpy as np

class ProbabilityDensityFunction(np.lib.mixins.NDArrayOperatorsMixin):
    """ 
    """

    def __init__(self, **kwargs):
        self._pdf_              = kwargs.get('probabilities', np.tile(0.5, kwargs.get('shape', (2, 2))))
        self._HANDLED_TYPES_    = (np.ndarray, type(self))
        if not np.allclose(self._pdf_.sum(), 1.0):
            self._pdf_ /= self._pdf_.sum()
            self._pdf_  = np.nan_to_num(self._pdf_)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            if not isinstance(x, self._HANDLED_TYPES_):
                return NotImplemented
        inputs = tuple(x._pdf_ if isinstance(x, ProbabilityDensityFunction) else x for x in inputs)
        if out:
            kwargs['out'] = (x._pdf_ if isinstance(x, ProbabilityDensityFunction) else x for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            return tuple(type(self)(probabilities = x) for x in result)
        elif method == 'at':
            return None
        else:
            return type(self)(probabilities = result)

    def __call__(self, **kwargs):
        return self._pdf_

    def __repr__(self):
        '''  '''
        repr_string = 'PDF: \n%s' % (
            self._pdf_.__str__()
        )
        return repr_string
    
pdf = ProbabilityDensityFunction()
print(np.subtract(pdf, pdf))