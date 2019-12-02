import numpy as np
from typing import Union
from math import e


def one_over_mu(x: float, div: bool = False) -> float:
    """
    A gaussian model for the magnetic permeability vs field of a permanent magnet
    :param div: To specify if derivative should be returned
    :param x: Norm of the magnetic field
    :return: Magnetic permeability
    """
    if div:
        return -(((-0.04416689710046033 * e ** (1.4181115724710933 * (-0.7754925571416307 + x) ** 2) * (
                -0.7754925571416307 + x)) /
                  -       (1. + e ** (1.4181115724710933 * (-0.7754925571416307 + x) ** 2)) ** 3 +
                  -      (0.04263697112349125 * (0.04885906655766546 + x)) / e ** (
                          42.077372984456105 * (0.04885906655766546 + x) ** 2)) /
                 -    (0.0005066496325619219 / e ** (42.077372984456105 * (0.04885906655766546 + x) ** 2) -
                       -       0.007786216888333133 / (
                               1. + e ** (1.4181115724710933 * (-0.7754925571416307 + x) ** 2)) ** 2) ** 2)

    return 1 / (-0.0005066496325619219 / e ** (
            42.077372984456105 * (0.04885906655766546 + x) ** 2) + 0.007786216888333133 / (
                        1. + e ** (1.4181115724710933 * (-0.7754925571416307 + x) ** 2)) ** 2)


if __name__ == '__main__':
    pass
