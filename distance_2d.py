#!/usr/bin/env python3
"""
2D Distance Calculation using Funshade
======================================

This script demonstrates how to use Funshade to securely compute the Euclidean distance
between two 2D coordinates and check if they are within a specified threshold.

The computation is done using secure multiparty computation (MPC) where:
- Party 0 (BP): Holds the first coordinate (x1, y1)
- Party 1 (Gate): Holds the second coordinate (x2, y2)
- Both parties jointly compute whether sqrt((x1-x2)² + (y1-y2)²) <= threshold without revealing their inputs

Usage:
    python distance_2d.py
"""

import math

import funshade
import numpy as np


def compute_2d_distance_secure(point1, point2, threshold, max_val=1000):
    """
    Securely compute whether the Euclidean distance between two 2D coordinates is within threshold.

    Args:
        point1 (tuple): First coordinate (x1, y1)
        point2 (tuple): Second coordinate (x2, y2)
        threshold (float): Distance threshold
        max_val (int): Maximum value for fixed-point scaling

    Returns:
        bool: True if sqrt((x1-x2)² + (y1-y2)²) <= threshold, False otherwise
    """

    x1, y1 = point1
    x2, y2 = point2

    # ==========================================================================#
    #                           PARAMETER SETUP                               #
    # ==========================================================================#
    K = 1  # Number of comparisons (just one pair of coordinates)
    l = 2  # Length of each vector (2D coordinates: x and y)

    # Convert to fixed-point integers for secure computation
    x1_fp = int(x1 * max_val)
    y1_fp = int(y1 * max_val)
    x2_fp = int(x2 * max_val)
    y2_fp = int(y2 * max_val)
    threshold_squared_fp = int(
        (threshold**2) * (max_val**2)
    )  # threshold² for squared distance

    print(f"Computing secure distance between {point1} and {point2}")
    print(f"Fixed-point values: point1=({x1_fp}, {y1_fp}), point2=({x2_fp}, {y2_fp})")
    print(f"Threshold squared (fixed-point): {threshold_squared_fp}")

    # Create the input vectors for distance computation
    # We'll compute the dot product of the two 2D vectors to get x1*x2 + y1*y2
    # Then use the formula: |p1-p2|² = |p1|² - 2*p1·p2 + |p2|²

    point1_vec = np.array(
        [x1_fp, y1_fp], dtype=funshade.DTYPE
    )  # First coordinate as vector
    point2_matrix = np.array(
        [[x2_fp, y2_fp]], dtype=funshade.DTYPE
    )  # Second coordinate as 1x2 matrix

    # Create parties
    class Party:
        def __init__(self, j: int):
            self.j = j

    BP = Party(0)  # Party 0 - holds point1
    Gate = Party(1)  # Party 1 - holds point2

    # ==========================================================================#
    #                            OFFLINE PHASE                                #
    # ==========================================================================#
    print("Setting up secure computation...")

    # Generate correlated randomness for secure computation
    d_x0, d_x1, d_y0, d_y1, d_xy0, d_xy1, r_in0, r_in1, k0, k1 = funshade.setup(
        K, l, threshold_squared_fp
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

    # Party 0 (BP) secret shares their coordinate (point1 -> Y in the protocol)
    BP.Y = point2_matrix.flatten()
    BP.D_y = funshade.share(K, l, BP.Y, BP.d_y)
    Gate.D_y = BP.D_y
    del BP.Y

    # ==========================================================================#
    #                             ONLINE PHASE                                #
    # ==========================================================================#
    print("Computing secure distance...")

    # Party 1 (Gate) secret shares their coordinate (point2 -> x in the protocol)
    Gate.x = np.tile(point1_vec, K)
    Gate.D_x = funshade.share(K, l, Gate.x, Gate.d_x)
    BP.D_x = Gate.D_x
    del Gate.x

    # Compute the masked dot product shares (this gives us x1*x2 + y1*y2)
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

    # Compute point1 · point2 (dot product result)
    dot_product = (BP.z_hat_j + Gate.z_hat_j) - (BP.r_in_j + Gate.r_in_j)

    # Compute squared distance: |point1 - point2|² = |point1|² - 2*point1·point2 + |point2|²
    point1_squared_norm = x1_fp * x1_fp + y1_fp * y1_fp
    point2_squared_norm = x2_fp * x2_fp + y2_fp * y2_fp
    squared_distance = point1_squared_norm - 2 * dot_product[0] + point2_squared_norm

    print(f"Computed squared distance: {squared_distance}")
    print(f"Threshold squared: {threshold_squared_fp}")

    # Check if distance is within threshold
    is_within_threshold = squared_distance <= threshold_squared_fp

    # ==========================================================================#
    #                          VERIFICATION                                    #
    # ==========================================================================#
    # Verify with ground truth
    actual_distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    actual_distance_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2
    expected_result = actual_distance <= threshold

    print(f"\nVerification:")
    print(f"Actual distance: {actual_distance:.6f}")
    print(f"Actual squared distance: {actual_distance_squared:.6f}")
    print(f"Calculated distance: {math.sqrt(squared_distance):.6f}")
    print(f"Calculated squared distance: {squared_distance:.6f}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Expected result: {expected_result}")
    print(f"Computed result: {is_within_threshold}")
    print(f"Results match: {expected_result == is_within_threshold}")

    return is_within_threshold


def main():
    """
    Example usage of secure 2D distance computation
    """
    print("=" * 60)
    print("Secure 2D Distance Computation with Funshade")
    print("=" * 60)

    # Example 1: Close points
    print("\nExample 1: Close points")
    point1 = (3.0, 4.0)
    point2 = (3.5, 4.2)
    threshold = 1.0  # Distance threshold
    result1 = compute_2d_distance_secure(point1, point2, threshold)
    actual_result1 = math.sqrt((3.0 - 3.5) ** 2 + (4.0 - 4.2) ** 2)
    # is the reult within the threshold of 5%?
    print(f"Actual distance: {actual_result1:.6f}")
    print(f"Computed result: {result1}")
    print(
        f"Results withing 5 percent margin: {(actual_result1 - threshold)/threshold <= 0.05}"
    )

    print("\n" + "-" * 40)

    # Example 2: Far points
    print("\nExample 2: Far points")
    point1 = (0.0, 0.0)
    point2 = (3.0, 4.0)  # Distance = 5.0
    threshold = 2.0
    result2 = compute_2d_distance_secure(point1, point2, threshold)

    print("\n" + "-" * 40)

    # Example 3: Negative coordinates
    print("\nExample 3: Negative coordinates")
    point1 = (-2.5, 1.5)
    point2 = (1.0, -2.0)
    threshold = 5.0
    result3 = compute_2d_distance_secure(point1, point2, threshold)

    print("\n" + "-" * 40)

    # Example 4: Points on unit circle
    print("\nExample 4: Points on unit circle")
    point1 = (1.0, 0.0)
    point2 = (0.0, 1.0)  # Distance = sqrt(2) ≈ 1.414
    threshold = 1.5
    result4 = compute_2d_distance_secure(point1, point2, threshold)

    print("\n" + "=" * 60)
    print("All computations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
