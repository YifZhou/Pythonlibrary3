#! /usr/bin/env python
"""use shelve to save parameters
"""


import shelve

def save(outputFN=None, **saveParams):
    if outputFN is None:
        print('No save file name given')
        return 1
    db = shelve.open(outputFN)
    for key in saveParams:
        db[key] = saveParams[key]
    db.close()
    return 0
