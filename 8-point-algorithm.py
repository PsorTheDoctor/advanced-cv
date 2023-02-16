import numpy as np
import pickle
from math import pi


def cross(t):
    return np.array([[0, -t[2], t[1]],
                     [t[2], 0, -t[0]],
                     [-t[1], t[0], 0]])


def triangulate(p, q, t, R):
    n = p.shape[1]
    assert n == q.shape[1], 'p and q must have the same number of columns.'
    P = np.zeros((3, n))

    i, j, k = R[0], R[2], R[2]
    kt = np.dot(k, t)
    proj = np.vstack((i, j))
    projt = np.dot(proj, t)

    C = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]]).astype(float)
    c = np.zeros(4)

    for m in range(n):
        C[:2, 2] = -p[:2, m]
        C[2:, :] = np.outer(q[:2, m], k) - proj
        c[2:] = kt * q[:2, m] - projt

        x = np.linalg.lstsq(C, c, rcond=None)
        P[:, m] = x[0]

    Q = np.dot(R, P - np.outer(t, np.ones((1, n))))
    return P, Q


def LonguetHiggins(p, q):
    """
    Reconstructs a scene from two projections.
    """
    n = p.shape[1]
    assert n == q.shape[1], 'p and q must have the same number of columns.'

    # Transforms images from 2D to 3D in the standard reference frame.
    ones = np.ones((1, n)).astype(float)
    p = np.concatenate((p, ones))
    q = np.concatenate((q, ones))

    # Set up matrix A such that A * E.flatten() = 0, where E is the esssential matrix.
    # The system encodes epipolar constraint q' * E * p = 0 for each of the points p and q.
    A = np.zeros((n, 9))
    for k in range(n):
        A[k, :] = np.outer(p[:, k], q[:, k]).flatten()

    assert np.linalg.matrix_rank(A) >= 8, 'Insufficient rank for A.'

    # The singular vector corresponding to the smallest singular value of A
    # is the arg_min_{norm(e) = 1} A * e, and is the LSE estimate of E.flatten()
    _, _, VT = np.linalg.svd(A)
    E = np.reshape(VT[-1, :], (3, 3), order='F')

    # Two possible translation vectors are t and -t, where t ia a unit vector in the null
    # space of E. The vector t (or -t) is also the second epipole of the camera pair.
    _, _, VET = np.linalg.svd(E)
    t = VET[2, :]

    # The cross-product matrix for vector t
    tx = cross(t)

    # Two rotation matrix choices are found by solving the Procrules problem
    # for the rows of E and tx, and allowing for the ambiguity resulting
    # from the sign of the null-space vectors (both E and tx are rank 2).
    # These two choices are independent of the sign of t, because both E and -E
    # are essential matrices.
    UF, _, VFT = np.linalg.svd(np.dot(E, tx))
    R1 = np.dot(UF, VFT)
    UF[:, 2] = -UF[:, 2]
    R2 = np.dot(UF, VFT)
    R2 *= np.linalg.det(R2)

    # Combine two sign options for t with the two choices for R
    tList = [t, t, -t, -t]
    RList = [R1, R2, R1, R2]

    # Pick the combination of t and R that yields the greatest number of
    # positive depth (Z) values in the structure results for the frames of
    # reference of both cameras. Ideally, all depth values should be positive.
    P, Q, npdMax = [], [], -1
    for k in range(4):
        tt, RR = tList[k], RList[k]
        PP, QQ = triangulate(p, q, tt, RR)
        npd = np.sum(np.logical_and(PP[2, :] > 0, QQ[2, :]))
        if npd > npdMax:
            t, R, P, Q, npdMax = tt, RR, PP, QQ, npd

    return t, R, P, Q


def load(filename):
    filename += '.pkl'
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def translationError(t, tTrue):
    t = t / np.linalg.norm(t)
    tTrue = tTrue / np.linalg.norm(tTrue)
    return np.arccos(np.dot(t, tTrue)) * 180.0 / pi


def rotationError(R, RTrue):
    def angle(R):
        U, _, VT = np.linalg.svd(R)
        R = np.dot(U, VT)

        # Sine and cosine of the rotation angle
        A = (R - R.T) / 2
        rho = np.array((A[2, 1], A[0, 2], A[1, 0]))
        sine = np.linalg.norm(rho)
        cosine = (np.trace(R) - 1) / 2
        theta = np.arctan2(sine, cosine)
        return theta * 180.0 / pi

    return np.abs(angle(np.dot(R, R.T)))


def structureError(P, PTrue):
    def normalize(P):
        centroid = np.mean(P, 1)
        Pc = P - np.outer(centroid, np.ones((P.shape[1])))
        scale = np.linalg.norm(Pc, ord='fro')  # Frobenius norm
        return Pc / scale

    P, PTrue = normalize(P), normalize(PTrue)
    M = np.dot(PTrue, P.T)
    U, _, VT = np.linalg.svd(M)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(np.dot(U, VT)))
    R = np.dot(U, np.dot(D, VT))

    P = np.dot(R, P)
    return 100 * np.linalg.norm(P - PTrue) / np.sqrt(P.shape[1])


def test(data):
    p, q = data['p'], data['q']
    t, R, P, Q = LonguetHiggins(p, q)
    sigma = data['sigma']
    tTrue, RTrue, PTrue = data['tTrue'], data['RTrue'], data['PTrue']
    et = translationError(t, tTrue)
    eR = rotationError(R, RTrue)
    eP = structureError(P, PTrue)
    print('Errors with noise sigma = ', sigma, ':', sep='')
    print('Translation {:.3g} degrees'.format(et))
    print('Rotation {:.3g} degrees'.format(eR))
    print('Structure {:.3g} percent of shape size'.format(eP))


clean = load('data/NoiseFree/NoiseFree')
noisy = load('data/Noisy/Noisy')
test(clean)
print()
test(noisy)
