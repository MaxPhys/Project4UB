import numpy as np
import matplotlib.pyplot as plt

# TASK 1

# Parameters, gamma fixed to 1
N = 10  # Number of particles in the box
D = 1  # Diffusion coefficient
tau = 1  # Relaxation time
eta = np.zeros((N, 2))  # Initial noise values
x = lambda t: np.random.normal(0, 1, size=(N, 2))  # Gaussian white noise
L = 10  # 2D LxL box size and initial conditions
r0 = np.random.uniform(-L / 2, L / 2, size=(N, 2))
t0 = 0
t_final = 10
dt = 0.01
num_steps = int((t_final - t0) / dt)

# Define the force function (zero for free particles, see task 2)
F = lambda r: np.zeros((N, 2))

# Increments of the Wiener process
dW_r = np.random.normal(0, np.sqrt(2 * dt), size=(num_steps, N, 2))
dW_eta = np.random.normal(0, np.sqrt(2 * D * dt / tau ** 2), size=(num_steps, N, 2))

# Initializing state vectors of particles
r = np.zeros((num_steps + 1, N, 2))
r[0] = r0
eta_vec = np.zeros((num_steps + 1, N, 2))
eta_vec[0] = eta

# Euler-Mayurama scheme
for i in range(num_steps):
    # Calculating increments for r(t)
    dr = F(r[i]) * dt + eta_vec[i] * np.sqrt(2 * dt) + dW_r[i]
    dr_norm = np.linalg.norm(dr, axis=1)
    theta = np.arctan2(dr[:, 1], dr[:, 0]) + np.random.normal(0, np.arctan(1), N)
    dx = dr_norm * np.cos(theta)
    dy = dr_norm * np.sin(theta)
    r[i + 1, :, 0] = r[i, :, 0] + dx
    r[i + 1, :, 1] = r[i, :, 1] + dy
    # For appearance reasons of the plot, PBC with numpy.clip() function
    r[i + 1] = np.clip(r[i + 1], -L / 2, L / 2)

    # Calculating the increments for eta(t)
    deta = -eta_vec[i] * dt / tau + x(i * dt) * np.sqrt(2 * D * dt / tau ** 2) + dW_eta[i]
    eta_vec[i + 1] = eta_vec[i] + deta

# Plots
fig, ax = plt.subplots()
for i in range(N):
    ax.plot(r[:, i, 0], r[:, i, 1], alpha=0.5)
ax.set_xlim(-L / 2, L / 2)
ax.set_ylim(-L / 2, L / 2)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

# TASK 2

F = lambda r: np.zeros((N, 2))

# Three different relaxation times
etas = [1, 10, 100]

# Initializing arrays
msd = np.zeros((len(etas), num_steps))
msd_free = np.zeros((len(etas), num_steps))

# Computing MSD for the three taus
for k, tau in enumerate(etas):
    eta = np.zeros((N, 2))
    dW_eta = np.random.normal(0, np.sqrt(2 * D * dt / tau ** 2), size=(num_steps, N, 2))
    r = np.zeros((num_steps + 1, N, 2))
    r[0] = r0
    eta_vec = np.zeros((num_steps + 1, N, 2))
    eta_vec[0] = eta
    for i in range(num_steps):
        dr = F(r[i]) * dt + eta_vec[i] * np.sqrt(2 * dt) + dW_r[i]
        dr_norm = np.linalg.norm(dr, axis=1)
        theta = np.arctan2(dr[:, 1], dr[:, 0]) + np.random.normal(0, np.arctan(1), N)
        dx = dr_norm * np.cos(theta)
        dy = dr_norm * np.sin(theta)
        r[i + 1, :, 0] = r[i, :, 0] + dx
        r[i + 1, :, 1] = r[i, :, 1] + dy
        r[i + 1] = np.clip(r[i + 1], -L / 2, L / 2) # numpy.clip(), as stated above

        deta = -eta_vec[i] * dt / tau + x(i * dt) * np.sqrt(2 * D * dt / tau ** 2) + dW_eta[i]
        eta_vec[i + 1] = eta_vec[i] + deta

    # Computing the MSD for respective tau
    for i in range(1, num_steps + 1):
        msd[k, i - 1] = np.mean(np.linalg.norm(r[i] - r0, axis=-1) ** 2)
        msd_free[k, i - 1] = 4 * D * i * dt

    # Respective resulting MSD
    print(f"MSD for tau={tau}: {msd[k, -1]}")

# Plot MSDs
fig, ax = plt.subplots()
for k, tau in enumerate(etas):
    ax.plot(np.arange(num_steps) * dt)
    plt.xlabel('num_steps')
    plt.ylabel('MSD')
plt.show()


# For comparing the results with the theoretical MSD, as given, define
def theoretical_msd(t, D, tau):
    if tau == 0:
        return 4 * D * t
    else:
        return 4 * D * (t + tau * (np.exp(-t / tau) - 1))


# MSD results for each tau
for k, tau in enumerate(etas):
    plt.plot(np.arange(num_steps)*dt, msd[k], label=f'tau={tau}')
    plt.plot(np.arange(num_steps)*dt, theoretical_msd(np.arange(num_steps)*dt, D, tau),
             linestyle='dashed', label=f'theoretical (tau={tau})')

plt.plot(np.arange(num_steps)*dt, msd_free[0], label='free')
plt.plot(np.arange(num_steps)*dt, theoretical_msd(np.arange(num_steps)*dt, D, 0),
         linestyle='dashed', label='theoretical (free)')
plt.xlabel('Time/s')
plt.ylabel('MSD')
plt.legend()
plt.show()

# TASK 3

relaxation_times = [1, 10, 100]
correlation_func = np.zeros((len(relaxation_times), num_steps))

for k, tau in enumerate(relaxation_times):
    eta = np.zeros((N, 2))
    dW_eta = np.random.normal(0, np.sqrt(2 * D * dt / tau ** 2), size=(num_steps, N, 2))
    eta_vec = np.zeros((num_steps + 1, N, 2))
    eta_vec[0] = eta
    for i in range(num_steps):
        deta = -eta_vec[i] * dt / tau + x(i * dt) * np.sqrt(2 * D * dt / tau ** 2) + dW_eta[i]
        eta_vec[i + 1] = eta_vec[i] + deta

    # Computing the correlation function
    for t in range(num_steps):
        for tp in range(num_steps - t):
            correlation_func[k, tp] += np.mean(np.sum(eta_vec[t] * eta_vec[t + tp], axis=1))

    correlation_func[k] /= N * num_steps

    # Plotting the correlation function
    plt.plot(np.arange(num_steps) * dt, correlation_func[k], label=f'tau={tau}')

    # Adding theoretical exponential decay line
    theoretical_decay = np.exp(-np.arange(num_steps) * dt / tau)
    plt.plot(np.arange(num_steps) * dt, theoretical_decay, linestyle='dashed', label='Theoretical')

plt.xlabel('t - t\'')
plt.ylabel('Correlation Function')
plt.legend()
plt.show()

# Initial conditions
etas = [10]  # relaxation times
forces = [0, 0.001, 0.01, 0.1, 1]  # force values
num_realizations = 10
r0 = np.random.uniform(-L / 2, L / 2, size=(N,))
t0 = 0
t_final = 10
dt = 0.01
num_steps = int((t_final - t0) / dt)

# Initialize arrays
msd = np.zeros((len(forces), num_steps))
displacement = np.zeros((len(forces), num_steps))


# Force function
def F(r):
    return np.tile(np.array([force, 0]), (N, 1))


# Simulations for each force
for k, force in enumerate(forces):
    # Average over all realizations
    for realization in range(num_realizations):
        # Initializing state vectors
        r = np.zeros((num_steps + 1, N, 2))
        r[0] = np.column_stack((r0, np.zeros_like(r0)))
        # Computing MSD and displacement for each time step
        for i in range(num_steps):
            # Increments for r(t)
            dr = (F(r[i]) + force * np.random.choice([-1, 1], size=(N, 2))) * dt \
                 + np.random.normal(0, np.sqrt(2 * D * dt), size=(N, 2))
            r[i + 1] = r[i] + dr
            r[i + 1] = np.clip(r[i + 1], -L / 2, L / 2)

        # MSD and displacement of respective realization
        for i in range(1, num_steps + 1):
            msd[k, i - 1] += np.mean(np.square(r[i, :, 0] - r0))
            displacement[k, i - 1] += np.mean(r[i, :, 0] - r0)

# Averaging  MSD and displacement over realizations
msd /= num_realizations
displacement /= num_realizations

# Plots in log-log
fig, axes = plt.subplots(2, 1)
fig.tight_layout(pad=4.0)

# Plot MSD
axes[0].loglog(np.arange(num_steps) * dt, msd.T)
axes[0].set_xlabel('Time/s')
axes[0].set_ylabel('MSD')

# Plot displacement
axes[1].loglog(np.arange(num_steps) * dt, displacement.T)
axes[1].set_xlabel('Time/s')
axes[1].set_ylabel('Displacement')

plt.show()

# TASK 4

# Parameters, gamma fixed to 1
eta = np.zeros((N, 2))
x = lambda t: np.random.normal(0, 1, size=(N, 2))
r0 = np.random.uniform(-L / 2, L / 2, size=(N, 2))
t0 = 0
t_final = 10
dt = 0.01
num_steps = int((t_final - t0) / dt)

# Constant force along y-axis
f_eps = [0, 0.001, 0.01, 0.1, 1]
displacement = np.zeros(len(f_eps))
msd_s = np.zeros(len(f_eps))

for f_index, f in enumerate(f_eps):
    dW_r = np.random.normal(0, np.sqrt(2 * dt), size=(num_steps, N, 2))

    # Initializing state vectors of particles
    r = np.zeros((num_steps + 1, N, 2))
    r[0] = r0
    eta_vec = np.zeros((num_steps + 1, N, 2))
    eta_vec[0] = eta

    # Euler-Mayurama scheme
    for i in range(num_steps):
        dr = (np.array([0, f]) * dt + eta_vec[i] * np.sqrt(2 * dt) + dW_r[i])
        dr_norm = np.linalg.norm(dr, axis=1)
        theta = np.arctan2(dr[:, 1], dr[:, 0]) + np.random.normal(0, np.arctan(1), N)
        dx = dr_norm * np.cos(theta)
        dy = dr_norm * np.sin(theta)
        r[i + 1, :, 0] = r[i, :, 0] + dx
        r[i + 1, :, 1] = r[i, :, 1] + dy
        r[i + 1] = np.clip(r[i + 1], -L / 2, L / 2)
        # Calculating increments for eta(t)
        deta = -eta_vec[i] * dt / tau + x(i * dt) * np.sqrt(2 * D * dt / tau ** 2) + dW_eta[i]
        eta_vec[i + 1] = eta_vec[i] + deta

    # Calculate MSD along Ox and displacement
    msd_s[f_index] = np.mean(np.mean((r[:, :, 1] - r0[:, 1]) ** 2, axis=1))
    displacement[f_index] = np.mean(np.mean(r[:, :, 1] - r0[:, 1], axis=1))

# MSD along Ox in log-log
fig, ax = plt.subplots()
ax.loglog(f_eps, msd_s, marker='o', linestyle='-', label='MSD along Ox')
ax.set_xlabel('Force')
ax.set_ylabel('MSD along Ox')
plt.legend()
plt.show()

# Displacement in log-log
fig, ax = plt.subplots()
ax.loglog(f_eps, displacement, marker='o', linestyle='-', label='Displacement')
ax.set_xlabel('Force')
ax.set_ylabel('Displacement')
plt.legend()
plt.show()

# Printing f where the force can be considered a perturbation
perturbation_threshold = 0.01
perturbation_index = np.argmax(displacement < perturbation_threshold)
perturbation_force = f_eps[perturbation_index]
print("The force can be considered a perturbation when f =", perturbation_force)

# Checking if linear response regime is reached
linear_response_threshold = 0.1
linear_response = np.all(displacement < linear_response_threshold)
if linear_response:
    print("We reach the linear response regime.")
else:
    print("We do not reach the linear response regime.")

# TASK 5

# Definitions
etas = [1, 10, 100]  # relaxation times
forces = [0, 0.001, 0.01, 0.1, 1]  # force values
num_realizations = 10

# Initializing arrays
msd = np.zeros((len(forces), num_steps))
displacement = np.zeros((len(forces), num_steps))

# Performing simulations for each force value
for k, force in enumerate(forces):
    # Compute average over multiple realizations
    for realization in range(num_realizations):
        # Initializing state vectors
        r = np.zeros((num_steps + 1, N, 2))
        r[0, :, 0] = r0[:, 0]  # Assigning x-coordinate values to the first column of r

        # MSD and displacement for each time step
        for i in range(num_steps):
            dr = (force * np.random.choice([-1, 1], size=(N, 2))) * dt \
                 + np.random.normal(0, np.sqrt(2 * D * dt), size=(N, 2))
            r[i + 1] = r[i] + dr
            r[i + 1] = np.clip(r[i + 1], -L / 2, L / 2)

        # MSD and displacement for the current realization
        for i in range(1, num_steps + 1):
            msd[k, i - 1] += np.mean(np.square(r[i, :, 0] - r0[:, 0].reshape(-1, 1)))
            displacement[k, i - 1] += np.mean(r[i, :, 0] - r0[:, 0].reshape(-1))

# Average MSD and displacement over realizations
msd /= num_realizations
displacement /= num_realizations

# Compute mu and D for each tau
mu_values = []
D_values = []

for tau in etas:
    # Exclude zero force values from calculations
    nonzero_forces = [force for force in forces if force != 0]
    num_nonzero_forces = len(nonzero_forces)

    # Calculating mu
    mu = displacement[:num_nonzero_forces, -1] / nonzero_forces
    mu_values.append(mu)

    # Calculating D
    D = msd[:num_nonzero_forces, -1] / (2 * (t_final - t0))
    D_values.append(D)


# mu vs tau
plt.figure()
for i, tau in enumerate(etas):
    plt.plot([etas[i]], [mu_values[i]], marker='o', linestyle='-', label=f'tau={etas[i]} & force={forces[i]}')
plt.xlabel('Relaxation time')
plt.ylabel('mu')
plt.title('mu vs tau')
plt.legend()
plt.show()

# D vs tau
plt.figure()
for i, tau in enumerate(etas):
    plt.plot([etas[i]], [D_values[i]], marker='o', linestyle='-', label=f'tau={etas[i]} & force={forces[i]}')
plt.xlabel('Relaxation time')
plt.ylabel('D')
plt.title('D vs tau')
plt.legend()
plt.show()

# TASK 7

# Initial conditions
N = 10
gamma = 1  # persistence length
D = 1
etas = [1, 10, 100]
forces = [0, 0.001, 0.01, 0.1, 1]
num_realizations = 10
L = 10
r0 = np.random.uniform(-L / 2, L / 2, size=(N,))
k = 1  # Stiffness constant
s0 = L / 2  # Center
t0 = 0
t_final = 10
dt = 0.01
num_steps = int((t_final - t0) / dt)

# Initializing arrays
msd = np.zeros((len(forces), num_steps))
displacement = np.zeros((len(forces), num_steps))

# Force function
def F(r):
    return -k * (s0 - r)

# Simulating for each force, same fashion as above
for k, force in enumerate(forces):
    for realization in range(num_realizations):
        r = np.zeros((num_steps + 1, N))
        r[0] = r0
        for i in range(num_steps):
            dr = (F(r[i]) + force * np.random.choice([-1, 1], size=N)) * dt / gamma + np.random.normal(0, np.sqrt(2 * D * dt), size=N)
            r[i + 1] = r[i] + dr
            r[i + 1] = np.clip(r[i + 1], -L / 2, L / 2)
        for i in range(1, num_steps + 1):
            msd[k, i - 1] += np.mean(np.square(r[i] - r0))
            displacement[k, i - 1] += np.mean(r[i] - r0)

# MSD and displacement over realizations
msd /= num_realizations
displacement /= num_realizations

# mu and D for each relaxation time
mu_values = []
D_values = []
Teff_T_values = []

for tau in etas:
    # Excluding zero force values from calculations
    nonzero_forces = [force for force in forces if force != 0]
    num_nonzero_forces = len(nonzero_forces)

    # mu
    mu = displacement[:num_nonzero_forces, -1] / nonzero_forces
    mu_values.append(mu)

    # D
    D = msd[:num_nonzero_forces, -1] / (2 * (t_final - t0))
    D_values.append(D)

    # Teff/T
    Teff_T = 1 / (1 + tau * k * mu)
    Teff_T_values.append(Teff_T)

# Plot mu vs tau
plt.figure()
for i, tau in enumerate(etas):
    plt.plot(forces[1:], mu_values[i], marker='o', linestyle='-', label=f'tau={tau}')
plt.xlabel('Force')
plt.ylabel('mu')
plt.title('mu vs tau')
plt.legend()
plt.show()

# Plot D vs tau
plt.figure()
for i, tau in enumerate(etas):
    plt.plot(forces[1:], D_values[i], marker='o', linestyle='-', label=f'tau={tau}')
plt.xlabel('Force')
plt.ylabel('D')
plt.title('D vs tau')
plt.legend()
plt.show()

# Plot Teff/T vs tau
plt.figure()
for i, tau in enumerate(etas):
    plt.plot(forces[1:], Teff_T_values[i], marker='o', linestyle='-', label=f'tau={tau}')
plt.xlabel('Force')
plt.ylabel('Teff/T')
plt.title('Teff/T vs tau')
plt.legend()
plt.show()
