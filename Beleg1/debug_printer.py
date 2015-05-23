#!/usr/bin/python

def get_debug_function(enabled=False):
    debugHeader = "{0} {1} {2}".format
    debugElements = "Element {n} of type {t} (Shape: {v.shape}):\n{v}".format
    def d(debugValues, caller=None):
        print( debugHeader("Debug message", caller if caller else "a", 30*"-") )
        print( "\n".join( [ debugElements(n=name, v=value, t=type(value)) for name, value in debugValues.iteritems() ] ) )

    if not enabled:
        return lambda debugValues, caller=None: 1+1
    else:
        return d
