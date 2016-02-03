import scipy.io as sio
from nipype.interfaces.base import Bunch


def get_conditions(path):

    mat = sio.loadmat(path)

    conditions = [Bunch(
        conditions=[i[0] for i in mat['names'][0]],
        onsets=[list(i[0]) if len(i[0]) > 1 else list(i.flatten()) for i in mat['onsets'][0]],
        durations=[list(i[0]) for i in mat['durations'][0]],
        pmod=[Bunch(
            name=list(i[list(mat['pmod'].dtype.names).index('name')][0][0]),
            param=[list(i[list(mat['pmod'].dtype.names).index('param')][0][0][0])],
            poly=list(i[list(mat['pmod'].dtype.names).index('poly')][0][0][0])
        ) for i in mat['pmod'][0]]
    )]

    return conditions

def make_contrast(name, test, convector, betas):
    return name, test, betas[:len(convector)], convector

def make_contrasts(contrasts, test, betas):
    return [(contrast[0], test, betas[:len(contrast[1])], contrast[1]) for contrast in contrasts]