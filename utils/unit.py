from astropy import units as u
import numpy as np

m=u.Unit(
    "m",

    doc="meter: base unit of length in SI"
)

mm=u.Unit(
    "mm",
    1.0e-3*m,
)

s=u.Unit(
    "s",

    doc="second: base unit of time in SI"         
)


rad=u.Unit(
    "rad",

    doc=(
        "radian: angular measurement of the ratio between the length "
        "on an arc and its radius"
    ),
)
deg=u.Unit(
    "deg",
    np.pi / 180.0 * rad,

    doc="degree: angular measurement 1/360 of full rotation",
)


kg=u.Unit(
    "kg",
    doc="kilogram: base unit of mass in SI.",
)
g=u.Unit(
    "g",
    1.0e-3 * kg,
)

N=u.Unit(
    "N",
    kg * m * s**-2,
    doc="Newton: force",
)

Pa=u.Unit(
    "Pa",
    N*m**-2,
    doc="Pascal: pressure"
)

MPa=u.Unit(
    "MPa",
    N*m**-2*1e6,
    doc="Pascal: pressure"
)

GPa=u.Unit(
    "GPa",
    N*m**-2*1e9,
    doc="Giga Pascal: pressure"
)


J=u.Unit(
    "J",
    N * m,
    doc="Joule: energy",
)

W=u.Unit(
    "W",
    J / s,
    doc="Watt: power",
)
