import numpy as np
from numpy import random

# Definitions -------------------------------------------------------------------------


def boxmuller(ndat, xnums, sigma):
    pi = np.pi
    k = 1
    while k < ndat-1:
        xi1 = np.random.rand()
        xi2 = np.random.rand()

        r = np.sqrt(-2 * np.log(xi1))
        phi = 2 * pi * xi2

        x1 = r * np.sin(phi)
        x2 = r * np.cos(phi)

        xnums[k] = sigma * x1
        xnums[k + 1] = sigma * x2
        k = k + 1
    return xnums

# -------------------------------------------------------------------------


def brownianmotion(N, L, dt, ntimes, D, tau, gamma, f, posx, posy, noisex, noisey, dx):
    sigma = 1
    i = 1
    while i < N:
        rand1 = np.random.rand()
        boxmuller(ntimes, gaussx, sigma)

        rand2 = np.random.rand()
        boxmuller(ntimes, gaussy, sigma)

        posx[i, 0] = L * rand1 - L / 2
        posy[i, 0] = L * rand2 - L / 2
        noisex[i, 0] = 0
        noisey[i, 0] = 0

        x1 = np.random.rand()
        if x1 < 0.5:
            eps = -1
        else:
            eps = 1

        dx[i, 0] = 0

        j = 1
        while j < ntimes:
            noisex[i, j] = noisex[i, j - 1] * (1-dt / tau) + (np.sqrt(2 * D * dt)/tau) * gaussx[j]
            noisey[i, j] = noisey[i, j - 1] * (1-dt / tau) + (np.sqrt(2 * D * dt)/tau) * gaussy[j]
            posx[i, j] = posx[i, j - 1] + (noisex[i, j]/gamma) * dt + (f/gamma) * dt
            posy[i, j] = posy[i, j - 1] + (noisey[i, j]/gamma) * dt

            dxpp[i, j] = eps * (posx[i, j] - posx[i, 0])
            # PBC (L x L box)
            if posx[i, j] > (L/2):
                posx[i, j] = posx[i, j] - L
            elif posx[i, j] < (-L/2):
                posx[i, j] = posx[i, j] + L
            elif posy[i, j] > (L/2):
                posy[i, j] = posy[i, j] - L
            elif posy[i, j] < (-L/2):
                posy[i, j] = posy[i, j] + L
            j = j + 1
        i = i + 1

    j = 0
    while j < ntimes:
        dx[j] = sum(dxpp[:, j])/N
        j = j + 1
    return posx, posy, noisex, noisey, dxpp

# -------------------------------------------------------------------------


def MSD(ntimes, N, posx, posy, msd, msdx):
    i, j = 1, 0
    while i < N:
        while j < ntimes:
            msdpp[i, j] = (posx[i, j] - posx[i, 0]) ** 2 + (posy[i, j] - posy[i, 0])** 2
            msdppx[i, j] = (posx[i, j] - posx[i, 0]) ** 2
            j = j + 1
        i = i + 1

    j = 0
    while j < ntimes:
        msd[j] = sum(msdpp[:, j])/N
        msdx[j] = sum(msdppx[:, j])/N
        j = j + 1
    return msdpp, msdppx, msd, msdx

# -------------------------------------------------------------------------


def correlation(N, ntimes, noisex, noisey, corr):
    i, j, k = 1, 0, 0
    while i < N:
        while j < ntimes:
            while k < ntimes:
                corrpp[i, j, k] = noisex[i, j] * noisex[i, k] + noisey[i, j] * noisey[i, k]
                k = k + 1
            j = j + 1
        i = i + 1

    i, j = 0, 0
    while i < ntimes:
        while j < ntimes:
            corr[i, j] = sum(corrpp[:, i, j])/N
            j = j + 1
        i = i + 1
    return corrpp, corr


# Start of the program
# Ex 1, 2, 3 --------------------------------------------------------------------------
gamma = 1
ntimes = 10
N = 10
L = 50
f = 0 #0.001, 0.01, 0.1, 1

posx = np.zeros([N, N])
posy = np.zeros([N, N])
noisex = np.zeros([N, ntimes])
noisey = np.zeros([N, ntimes])
corr = np.zeros([ntimes, ntimes])
msd, dx, msdx, avdx, avmsdx = np.zeros(ntimes), np.zeros([N, ntimes]), np.zeros(ntimes), np.zeros(ntimes), np.zeros(ntimes)

dxpp = np.zeros([N, ntimes])
msdpp = np.zeros([N, ntimes])
msdppx = np.zeros([N, ntimes])
corrpp = np.zeros([ntimes, ntimes, ntimes])

gaussx = np.zeros(ntimes)
gaussy = np.zeros(ntimes)

tau = 1 # 10, 100
dt = tau/100
D = tau

brownianmotion(N, L, dt, ntimes, D, tau, gamma, f, posx, posy, noisex, noisey, dx)
MSD(ntimes, N, posx, posy, msd, msdx)
correlation(N, ntimes, noisex, noisey, corr)

j = 0
while j < ntimes:
    print("tau: ", tau)
    print("j * dt: ", j * dt)
    print("msd[j]: ", msd[j])
    j = j + 1


j, k = 0, 0
while j < ntimes:
    while k < ntimes:
        print("tau: ", tau)
        print("i - j: ", i - j)
        print("corr[i, j]: ", corr[i, j])
        k = k + 1
    j = j + 1


# Ex 4 --------------------------------------------------------------------------------
tau = 10
dt = 0.1
ntimes = 10

allocate(avdx[0: ntimes])
allocate(avmsdx[0: ntimes])


i, k = -3, 1
while i < 0:
    f = 10 ** dble(i)
    D = 1
    avmsdx = 0
    avdx = 0
    i = i + 1

    brownianmotion(N, L, dt, ntimes, D, tau, gamma, f, posx, posy, noisex, noisey, dx)
    MSD(ntimes, N, posx, posy, msd, msdx)
    avmsdx[:] = avmsdx[:] + msdx[:]
    avdx[:] = avdx[:] + dx[:]

j = 0
while j < ntimes:
    print("f: ", f)
    print("j * dt: ", j * dt)
    print("avmsdx: ", avmsdx[j]/10)
    print("avdx: ", avdx[j]/10)
    j = j + 1

# Ex 5 --------------------------------------------------------------------------------
i = 1
while i < 100:
    tau = dble(i)
    dt = tau / 100
    i = i + 1
