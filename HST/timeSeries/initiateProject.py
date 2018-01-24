#! /usr/bin/env python3
import configparser
import glob
import os
import shutil
from os import path
from HST.timeSeries import timeSeriesInfo


"""initiate the directory for a set of HST time-resolved observations
"""


def initiateProject(configFN):
    """initiate the environment for a set of HST time-resolved observations

    :param configFN: configuration file name
    """
    conf = configparser.ConfigParser()
    conf.read(configFN)
    projName = conf['general']['projName']
    projDIR = path.expanduser(conf['general']['projDIR'])
    dataDIR = path.expanduser(conf['general']['dataDIR'])
    fileType = conf['general']['fileType']
    # test if the project directory already exists
    if path.exists(projDIR):
        cont = input("Directory {0} already exists. Press [y] to continue".format(projDIR))
        if cont.lower() == 'y':
            # clean the directory
            shutil.rmtree(projDIR)
        else:
            exit(1)
    # first establish the directory tree
    DIRList = ['code', 'info', 'data', 'save', 'result', 'plot', 'calibration']
    os.makedirs(projDIR)
    for directory in DIRList:
        print('Making directory {0}'.format(path.join(projDIR, directory)))
        os.makedirs(path.join(projDIR, directory))

    # copy data from dataDIR to target directory
    fnList = glob.glob(path.join(dataDIR, '*{0}.fits'.format(fileType)))
    for fn in fnList:
        basename = path.basename(fn)
        print('Copying file {0}'.format(basename))
        shutil.copyfile(fn, path.join(projDIR, 'data', basename))

    # create info file
    dataInfo = timeSeriesInfo(fileType, path.join(projDIR, 'data'))
    dataInfoName = projName + '_fileInfo.csv'
    dataInfo.to_csv(path.join(projDIR, 'info', dataInfoName), index=False)
    # write configuration file to project directory
    configFNBasename = path.basename(configFN)
    conf['general']['infofn'] = path.join(projDIR, 'info', dataInfoName)
    with open(path.join(projDIR, 'info', configFNBasename), 'w') as configWFN:
        conf.write(configWFN)
    print('Project {0} initiated'.format(projName))
    print('Project established at {0}'.format(projDIR))


if __name__ == '__main__':
    configFN = '/Users/ZhouYf/Documents/temp/test.ini'
    initiateProject(configFN)
