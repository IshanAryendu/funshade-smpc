#!/usr/bin/env python3
"""
2D Angle Calculation using Funshade
===================================

This script demonstrates how to use Funshade to securely compute the angle
between two 2D coordinates.

Usage:
    python angle_2d.py
"""

import math
import funshade
import numpy as np

def compute_2d_angle_secure(point1, point2, max_val=1000):
    """
    Securely compute the angle (in radians) between two 2D coordinates.

    Args:
        point1 (tuple): First coordinate (x1, y1)
        point2 (tuple): Second coordinate (x2, y2)
        max_val (int): Maximum value for fixed-point scaling

    Returns:
        float: Angle in radians between the two points
    """

    x1, y1 = point1
    x2, y2 = point2

    # Convert to fixed-point integers for secure computation
    x1_fp = int(x1 * max_val)
    y1_fp = int(y1 * max_val)
    x2_fp = int(x2 * max_val)
    y2_fp = int(y2 * max_val)

    print(f"Computing secure angle between {point1} and {point2}")
    print(f"Fixed-point values: point1=({x1_fp}, {y1_fp}), point2=({x2_fp}, {y2_fp})")

    # Prepare vectors
    vec1 = np.array([x1_fp, y1_fp], dtype=funshade.DTYPE)
    vec2 = np.array([x2_fp, y2_fp], dtype=funshade.DTYPE)

    # Create parties
    class Party:
        def __init__(self, j: int):
            self.j = j

    BP = Party(0)
    Gate = Party(1)

    K = 1
    l = 2

    # Setup phase (simulate randomness for secure computation)
    d_x0, d_x1, d_y0, d_y1, d_xy0, d_xy1, r_in0, r_in1, k0, k1 = funshade.setup(K, l, 0)

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

    # Secret share vectors
    BP.Y = vec2
    BP.D_y = funshade.share(K, l, BP.Y, BP.d_y)
    Gate.D_y = BP.D_y
    del BP.Y

    Gate.x = np.tile(vec1, K)
    Gate.D_x = funshade.share(K, l, Gate.x, Gate.d_x)
    BP.D_x = Gate.D_x
    del Gate.x

    # Secure dot product
    BP.z_hat_j = funshade.eval_dist(K, l, BP.j, BP.r_in_j, BP.D_x, BP.D_y, BP.d_x_j, BP.d_y_j, BP.d_xy_j)
    Gate.z_hat_j = funshade.eval_dist(K, l, Gate.j, Gate.r_in_j, Gate.D_x, Gate.D_y, Gate.d_x_j, Gate.d_y_j, Gate.d_xy_j)

    # Exchange shares to reconstruct dot product
    dot_product = (BP.z_hat_j + Gate.z_hat_j) - (BP.r_in_j + Gate.r_in_j)
    dot_product_val = dot_product[0]

    # Compute norms
    norm1_sq = x1_fp * x1_fp + y1_fp * y1_fp
    norm2_sq = x2_fp * x2_fp + y2_fp * y2_fp
    norm1 = math.sqrt(norm1_sq)
    norm2 = math.sqrt(norm2_sq)

    # Compute angle
    if norm1 == 0 or norm2 == 0:
        print("One of the vectors is zero; angle is undefined.")
        return None

    cos_theta = dot_product_val / (norm1 * norm2)
    # Clamp to [-1, 1] to avoid domain errors
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle = math.acos(cos_theta)

    print(f"Dot product: {dot_product_val}")
    print(f"Norm1: {norm1}, Norm2: {norm2}")
    print(f"cos(theta): {cos_theta}")
    print(f"Angle (radians): {angle}")
    print(f"Angle (degrees): {math.degrees(angle)}")

    # Verification
    actual_dot = x1 * x2 + y1 * y2
    actual_norm1 = math.sqrt(x1 * x1 + y1 * y1)
    actual_norm2 = math.sqrt(x2 * x2 + y2 * y2)
    if actual_norm1 == 0 or actual_norm2 == 0:
        actual_angle = None
    else:
        actual_cos = actual_dot / (actual_norm1 * actual_norm2)
        actual_cos = max(-1.0, min(1.0, actual_cos))
        actual_angle = math.acos(actual_cos)
    print(f"Actual angle (radians): {actual_angle}")
    print(f"Results match: {abs(angle - actual_angle) < 0.05 if actual_angle is not None else 'N/A'}")

    return angle

def main():
    print("=" * 60)
    print("Secure 2D Angle Computation with Funshade")
    print("=" * 60)

    # Example 1: 45 degrees
    print("\nExample 1: 45 degrees")
    point1 = (1.0, 0.0)
    point2 = (1.0, 1.0)
    compute_2d_angle_secure(point1, point2)

    print("\n" + "-" * 40)

    # Example 2: 90 degrees
    print("\nExample 2: 90 degrees")
    point1 = (1.0, 0.0)
    point2 = (0.0, 1.0)
    compute_2d_angle_secure(point1, point2)

    print("\n" + "-" * 40)

    # Example 3: 0 degrees (same direction)
    print("\nExample 3: 0 degrees")
    point1 = (2.0, 2.0)
    point2 = (4.0, 4.0)
    compute_2d_angle_secure(point1, point2)

    print("\n" + "-" * 40)

    # Example 4: 180 degrees (opposite direction)
    print("\nExample 4: 180 degrees")
    point1 = (1.0, 0.0)
    point2 = (-1.0, 0.0)
    compute_2d_angle_secure(point1, point2)

    print("\n" + "=" * 60)
    print("All computations completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()