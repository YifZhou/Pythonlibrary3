#! /usr/bin/env python
"""use pickle to save parameters
"""


# import shelve
import pickle

def save(outputFN=None, **saveParams):
    if outputFN is None:
        print('No save file name given')
        return 1
    # db = shelve.open(outputFN)

    # for key in saveParams:
    #     db[key] = saveParams[key]
    # db.close()
    with open(outputFN, 'wb') as pkl:
        pickle.dump(saveParams, pkl)
    return 0
