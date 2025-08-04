#!/usr/bin/env python3
"""
1D Distance Calculation using Funshade
======================================

This script demonstrates how to use Funshade to securely compute the distance
between two 1D coordinates and check if they are within a specified threshold.

The computation is done using secure multiparty computation (MPC) where:
- Party 0 (BP): Holds the first coordinate
- Party 1 (Gate): Holds the second coordinate
- Both parties jointly compute whether |x1 - x2|^2 <= threshold without revealing their inputs

Usage:
    python distance_1d.py
"""

import funshade
import numpy as np


def compute_1d_distance_secure(x1, x2, threshold, max_val=1000):
    """
    Securely compute whether the distance between two 1D coordinates is within threshold.

    Args:
        x1 (float): First coordinate
        x2 (float): Second coordinate
        threshold (float): Distance threshold
        max_val (int): Maximum value for fixed-point scaling

    Returns:
        bool: True if |x1 - x2|^2 <= threshold, False otherwise
    """

    # ==========================================================================#
    #                           PARAMETER SETUP                               #
    # ==========================================================================#
    K = 1  # Number of comparisons (just one pair of coordinates)
    l = 1  # Length of each vector (1D coordinates)

    # Convert to fixed-point integers for secure computation
    x1_fp = int(x1 * max_val)
    x2_fp = int(x2 * max_val)
    threshold_fp = int(threshold * (max_val**2))  # threshold for squared distance

    print(f"Computing secure distance between {x1} and {x2}")
    print(f"Fixed-point values: x1={x1_fp}, x2={x2_fp}, threshold={threshold_fp}")

    # Create the input vectors for distance computation
    # For distance |x1 - x2|^2, we compute (x1 - x2) * (x1 - x2)
    # This can be rewritten as: x1^2 - 2*x1*x2 + x2^2
    # We'll use the dot product to compute x1*x2, then add x1^2 + x2^2 separately

    x = np.array([x1_fp], dtype=funshade.DTYPE)  # First coordinate
    Y = np.array([[x2_fp]], dtype=funshade.DTYPE)  # Second coordinate as 1x1 matrix

    # Create parties
    class Party:
        def __init__(self, j: int):
            self.j = j

    BP = Party(0)  # Party 0 - holds x1
    Gate = Party(1)  # Party 1 - holds x2

    # ==========================================================================#
    #                            OFFLINE PHASE                                #
    # ==========================================================================#
    print("Setting up secure computation...")

    # Generate correlated randomness for secure computation
    d_x0, d_x1, d_y0, d_y1, d_xy0, d_xy1, r_in0, r_in1, k0, k1 = funshade.setup(
        K, l, threshold_fp
    )

    # Distribute randomness to parties
    BP.d_x_j = d_x0
    Gate.d_x_j = d_x1
    BP.d_y_j = d_y0
    Gate.d_y_j = d_y1
    BP.d_xy_j = d_xy0
    Gate.d_xy_j = d_xy1
    BP.r_in_j = r_in0
    Gate.r_in_j = r_in1
    BP.k_j = k0
    Gate.k_j = k1
    BP.d_y = d_y0 + d_y1
    Gate.d_x = d_x0 + d_x1

    # Party 0 (BP) secret shares their coordinate (x1 -> Y in the protocol)
    BP.Y = Y.flatten()
    BP.D_y = funshade.share(K, l, BP.Y, BP.d_y)
    Gate.D_y = BP.D_y
    del BP.Y

    # ==========================================================================#
    #                             ONLINE PHASE                                #
    # ==========================================================================#
    print("Computing secure distance...")

    # Party 1 (Gate) secret shares their coordinate (x2 -> x in the protocol)
    Gate.x = np.tile(x, K)
    Gate.D_x = funshade.share(K, l, Gate.x, Gate.d_x)
    BP.D_x = Gate.D_x
    del Gate.x

    # Compute the masked dot product shares (this gives us x1*x2)
    BP.z_hat_j = funshade.eval_dist(
        K, l, BP.j, BP.r_in_j, BP.D_x, BP.D_y, BP.d_x_j, BP.d_y_j, BP.d_xy_j
    )
    Gate.z_hat_j = funshade.eval_dist(
        K,
        l,
        Gate.j,
        Gate.r_in_j,
        Gate.D_x,
        Gate.D_y,
        Gate.d_x_j,
        Gate.d_y_j,
        Gate.d_xy_j,
    )

    # Exchange shares to reconstruct the dot product
    BP.z_hat_nj = Gate.z_hat_j
    Gate.z_hat_nj = BP.z_hat_j

    # Compute x1*x2 (dot product result)
    dot_product = (BP.z_hat_j + Gate.z_hat_j) - (BP.r_in_j + Gate.r_in_j)

    # Compute squared distance: |x1 - x2|^2 = x1^2 - 2*x1*x2 + x2^2
    x1_squared = x1_fp * x1_fp
    x2_squared = x2_fp * x2_fp
    squared_distance = x1_squared - 2 * dot_product[0] + x2_squared

    print(f"Computed squared distance: {squared_distance}")
    print(f"Threshold: {threshold_fp}")

    # Check if distance is within threshold
    is_within_threshold = squared_distance <= threshold_fp

    # ==========================================================================#
    #                          VERIFICATION                                    #
    # ==========================================================================#
    # Verify with ground truth
    actual_distance_squared = (x1 - x2) ** 2
    expected_result = actual_distance_squared <= threshold

    print(f"\nVerification:")
    print(f"Actual distance: {abs(x1 - x2):.6f}")
    print(f"Actual squared distance: {actual_distance_squared:.6f}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Expected result: {expected_result}")
    print(f"Computed result: {is_within_threshold}")
    print(f"Results match: {expected_result == is_within_threshold}")

    return is_within_threshold


def main():
    """
    Example usage of secure 1D distance computation
    """
    print("=" * 60)
    print("Secure 1D Distance Computation with Funshade")
    print("=" * 60)

    # Example 1: Close coordinates
    print("\nExample 1: Close coordinates")
    x1, x2 = 5.2, 5.7
    threshold = 1.0  # Distance threshold
    result1 = compute_1d_distance_secure(x1, x2, threshold)

    print("\n" + "-" * 40)

    # Example 2: Far coordinates
    print("\nExample 2: Far coordinates")
    x1, x2 = 1.0, 8.0
    threshold = 2.0
    result2 = compute_1d_distance_secure(x1, x2, threshold)

    print("\n" + "-" * 40)

    # Example 3: Negative coordinates
    print("\nExample 3: Negative coordinates")
    x1, x2 = -3.5, 2.1
    threshold = 5.0
    result3 = compute_1d_distance_secure(x1, x2, threshold)

    print("\n" + "=" * 60)
    print("All computations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
