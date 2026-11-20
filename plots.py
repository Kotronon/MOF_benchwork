import numpy as np
import matplotlib.pyplot as plt

base = "outputs/module_D_mil53_co2/runs/classical_isotherm_6p_800cycles_v1/jobs/hybrid_npt__p004__seed12345/results/hybrid_npt__p004__seed12345"
m = np.loadtxt(f"{base}/measurements.txt", comments="#")
c = np.loadtxt(f"{base}/cell_history.txt", comments="#")

cycle = np.arange(1, len(m) + 1)
a = c[:, 1]
b = np.sqrt(c[:, 2]**2 + c[:, 4]**2)
cc = np.sqrt(c[:, 3]**2 + c[:, 5]**2 + c[:, 6]**2)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(cycle, m[:, 1])
ax[0].set_ylabel("CO₂-Beladung (mol/kg)")
ax[1].plot(cycle, a, label="a")
ax[1].plot(cycle, b, label="b")
ax[1].plot(cycle, cc, label="c")
ax[1].set(xlabel="Zyklus", ylabel="Zellvektorlänge (Å)")
ax[1].legend()
plt.show()