import warnings

import numpy as np
import pint

ureg = pint.UnitRegistry()


def bcs_gap(T, Tc, Delta0, a):
    kBTc_over_Delta = (ureg("k_B") * Tc / Delta0).to_base_units().m
    return Delta0 * np.tanh(np.pi * kBTc_over_Delta * np.sqrt(a * (Tc / T - 1)))


def Vc_KO(
    T: float, phi: float, D: float = 1.0, Tc: float = 1.95, a: float = 1.0
) -> pint.Quantity:
    T = T * ureg("K")
    Tc = Tc * ureg("K")
    Delta0 = 1.76 * ureg("k_B") * Tc
    Delta = bcs_gap(T, Tc, Delta0, a)
    fact = np.sqrt(1 - D * np.sin(phi / 2) ** 2)
    tanh = np.tanh((Delta / (2 * ureg("k_B") * T)).to_base_units().m * fact)
    Vc = (np.pi * Delta / (2 * ureg("e")) * np.sin(phi) / fact * tanh).to("mV")
    return Vc


def get_Vc(
    kBT_over_J0: float,
    Tc: float,
    kBTc_over_J0: float,
    transparency: float,
    nphis: int = 201,
) -> pint.Quantity:
    phis = np.linspace(0, np.pi, nphis)
    T = kBT_over_J0 / kBTc_over_J0 * Tc
    if T >= Tc:
        return 0.0
    if transparency == 0:
        Vc = Vc_KO(T, np.pi / 2, D=transparency, Tc=Tc).magnitude
    else:
        Vcs = [Vc_KO(T, phi, D=transparency, Tc=Tc) for phi in phis]
        Vc = max(Vcs).magnitude
    return Vc


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kBT-over-J0", type=float)
    parser.add_argument("--Tc", type=float, default=1.95)
    parser.add_argument("--kBTc-over-J0", type=float, default=1.5)
    parser.add_argument("--transparency", type=float, default=1.0)

    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore")

        Vc0 = get_Vc(1e-6, args.Tc, args.kBTc_over_J0, args.transparency)
        Vc = get_Vc(args.kBT_over_J0, args.Tc, args.kBTc_over_J0, args.transparency)

        print(Vc.magnitude / Vc0.magnitude)


if __name__ == "__main__":
    main()
