import warnings

import numpy as np
import pint

ureg = pint.UnitRegistry()


def bcs_gap(T: float, Tc: float = 1.0, Delta0: float = 1.764, a: float = 1.0):
    return Delta0 * np.tanh(np.pi / Delta0 * np.sqrt(a * (Tc / T - 1)))


def Vc_KO(T: float, phi: float, D: float = 1.0, Tc: float = 1.95) -> float:
    T = T * ureg("K")
    Tc = Tc * ureg("K")
    Delta = bcs_gap(T.magnitude, Tc=Tc.magnitude, Delta0=1.764) * ureg("k_B") * Tc
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
) -> float:
    phis = np.linspace(0, np.pi, nphis)
    T = kBT_over_J0 / kBTc_over_J0 * Tc
    if T >= Tc:
        return 0.0

    Vcs = [Vc_KO(T, phi, D=transparency, Tc=Tc).magnitude for phi in phis]
    Vc = max(Vcs)
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

        print(Vc / Vc0)


if __name__ == "__main__":
    main()
