import numpy as np
from typing import Union
from math import exp


def mu(b: float, div: bool = False) -> float:
    """
    A gaussian model for the magnetic permeability vs field of a permanent magnet
    :param div: To specify if derivative should be returned
    :param b: Norm of the magnetic field
    :return: Magnetic permeability
    """
    if div:
        return -0.000506649632561922 * (-84.1547459689122 * b - 4.11172233443851) * exp(
            -42.0773729844561 * (b + 0.0488590665576655) ** 2) - 0.0155724337766663 * (
                           2.83622314494219 * b - 2.19946993929549) * exp(
            1.41811157247109 * (b - 0.775492557141631) ** 2) / (
                           exp(1.41811157247109 * (b - 0.775492557141631) ** 2) + 1) ** 3

    return -0.000506649632561922 * exp(-42.0773729844561 * (b + 0.0488590665576655) ** 2) + 0.00778621688833313 / (
                exp(1.41811157247109 * (b - 0.775492557141631) ** 2) + 1) ** 2

