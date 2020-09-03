""" Top-level module for hessian eigenvec computation """
from .power_iter import power_iteration, deflated_power_iteration
from .lanczos import lanczos
from .hvp_operator import HVPOperator, compute_hessian_eigenthings
from .hvp_operator_layer import HVPOperator_layer, compute_hessian_eigenthings_layer

__all__ = [
    "power_iteration",
    "deflated_power_iteration",
    "lanczos",
    "HVPOperator",
    "compute_hessian_eigenthings",
    "HVPOperator_layer",
    "compute_hessian_eigenthings_layer"
]

name = "hessian_eigenthings"
