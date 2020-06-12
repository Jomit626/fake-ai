from numba import jitclass, typeof, njit

import torch
import os
version = "0.0.0"

trace_id = {
    'CNHKG-MXZLO':0,

    'CNSHA-PAMIT':1,
    'CNSHA-SGSIN':2,

    'CNSHK-CLVAP':3,
    'CNSHK-ESALG':4,
    'CNSHK-GRPIR':5,
    'CNSHK-LBBEY':6,
    'CNSHK-MYTPP':7,
    'CNSHK-PKQCT':8,
    'CNSHK-SGSIN':9,
    'CNSHK-SIKOP':10,
    'CNSHK-ZADUR':11,

    'CNYTN-ARENA':12,
    'CNYTN-CAVAN':13,
    'CNYTN-MATNG':14,
    'CNYTN-MTMLA':15,
    'CNYTN-MXZLO':16,
    'CNYTN-NZAKL':17,
    'CNYTN-PAONX':18,
    'CNYTN-RTM'  :19,

    'COBUN-HKHKG':20,

    'HKHKG-FRFOS':21,
}
