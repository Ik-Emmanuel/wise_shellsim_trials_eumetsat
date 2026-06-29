"""A/B test: isolate whether over-scaling comes from model CONFIG or Chl INPUT.
Constant forcing, 365 daily forward-Euler steps, same seed in both configs."""
import os, contextlib
import numpy as np
import pyfabm

N = 365
T_CONST, S_CONST = 10.0, 26.0

FULL = "single_timeseries/fabm.yaml"                       # Chl + POC + POM
PARTIAL = "notebook_dropdowns/20260625/partial_fabm.yaml"  # Chl only


def run(cfg, chl, poc=None, pom=None, tpm=None):
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        m = pyfabm.Model(cfg)
        m.cell_thickness = 1.0
        m.dependencies["seeding_rate"].value = 0.0
        m.dependencies["harvest_ratio"].value = 0.0
        m.dependencies["current_speed"].value = 1.0
        m.dependencies["air_exposure"].value = 0.0
        m.dependencies["temperature"].value = T_CONST
        m.dependencies["practical_salinity"].value = S_CONST
        m.dependencies["number_of_days_since_start_of_the_year"].value = 1.0
        m.findStateVariable("Chl1/Chl").value = chl
        if poc is not None: m.findStateVariable("POC1/POC").value = poc
        if pom is not None: m.findStateVariable("POM1/POM").value = pom
        if tpm is not None:
            try: m.findStateVariable("TPM1/TPM").value = tpm
            except Exception: pass
        assert m.start()
        for _ in range(N):
            m.dependencies["temperature"].value = T_CONST
            m.dependencies["practical_salinity"].value = S_CONST
            m.findStateVariable("Chl1/Chl").value = chl
            if poc is not None: m.findStateVariable("POC1/POC").value = poc
            if pom is not None: m.findStateVariable("POM1/POM").value = pom
            m.state[:] += m.getRates() * 86400.0
        return (m.diagnostic_variables["Oyster/TFW"].value,
                m.diagnostic_variables["Oyster/Shell_Length"].value,
                float(m.state[0]))


print(f"{'scenario':48s} {'TFW(g)':>10s} {'SL(cm)':>8s} {'STE(J)':>12s}")
print("-" * 82)
print("[PARTIAL = partial_fabm.yaml, Chl-only, what the gridded notebook actually uses]")
for chl in (1.0, 3.5, 9.0, 13.0, 30.0, 100.0):
    r = run(PARTIAL, chl=chl)
    print(f"{'  PARTIAL  Chl=%-6.1f' % chl:48s} {r[0]:10.2f} {r[1]:8.2f} {r[2]:12.1f}")
print("[FULL = single-timeseries fabm.yaml, Chl+POC(1400)+POM(40)]")
for chl in (1.0, 3.5, 9.0, 13.0):
    r = run(FULL, chl=chl, poc=1400.0, pom=40.0, tpm=100.0)
    print(f"{'  FULL     Chl=%-6.1f' % chl:48s} {r[0]:10.2f} {r[1]:8.2f} {r[2]:12.1f}")
