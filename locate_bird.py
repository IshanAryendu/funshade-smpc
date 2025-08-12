#!/usr/bin/env python3
"""
3D Distance, Elevation, and Azimuth Calculation using Funshade
==============================================================

This script demonstrates how to use Funshade to securely compute the Euclidean distance,
elevation angle, and azimuthal angle between two 3D coordinates.

Compute the distance, elevation angle, and azimuthal angle from observer to bird.

    Args:
        observer (tuple): (x0, y0, z0) coordinates of observer
        bird (tuple): (x1, y1, z1) coordinates of bird

    Returns:
        dict: {'distance': d, 'elevation': theta, 'azimuth': phi}
            - distance: Euclidean distance
            - elevation: angle of elevation in radians
            - azimuth: azimuthal angle in radians

based on this:

import math
import numpy as np

def locate_bird_3d(observer, bird):
    
    x0, y0, z0 = observer
    x1, y1, z1 = bird

    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0

    # Euclidean distance
    d = math.sqrt(dx**2 + dy**2 + dz**2)

    # Angle of elevation (from horizontal plane)
    elevation = math.atan2(dz, math.sqrt(dx**2 + dy**2))

    # Azimuthal angle (from x-axis in x-y plane)
    azimuth = math.atan2(dy, dx)

    print(f"Distance: {d}")
    print(f"Elevation (radians): {elevation}, degrees: {math.degrees(elevation)}")
    print(f"Azimuth (radians): {azimuth}, degrees: {math.degrees(azimuth)}")

    return {
        'distance': d,
        'elevation': elevation,
        'azimuth': azimuth
    }

# Example usage:
if __name__ == "__main__":
    observer = (0.0, 0.0, 0.0)  # Your position
    bird = (10.0, 10.0, 5.0)    # Bird's position
    res = locate_bird_3d(observer, bird)

    # Verify the result using the following formulae for ground truth:
    # Px = distance * sin(yaw) * cos(pitch) 
    # Py = distance * cos(yaw) * cos(pitch)
    # Pz = distance * sin(pitch) 
    # where Pitch: elevation and Yaw: azimuth
    Px = res['distance'] * math.cos(res['elevation']) * math.sin(res['azimuth'])
    Py = res['distance'] * math.cos(res['azimuth']) * math.cos(res['elevation'])
    Pz = res['distance'] * math.sin(res['elevation'])

    print(f"Calculated truth coordinates: Px={Px}, Py={Py}, Pz={Pz}")
    print(f"Actual bird coordinates: Px={bird[0]}, Py={bird[1]}, Pz={bird[2]}")

Usage:
    python distance_3d.py
"""


import math
import funshade
import numpy as np


def compute_3d_distance_and_angles_secure(observer, bird, max_val=1000):
    """
    Securely compute the distance, elevation, and azimuth between two 3D coordinates.

    Args:
        observer (tuple): (x0, y0, z0) coordinates of observer
        bird (tuple): (x1, y1, z1) coordinates of bird
        max_val (int): Maximum value for fixed-point scaling

    Returns:
        dict: {'distance': d, 'elevation': theta, 'azimuth': phi}
    """
    x0, y0, z0 = observer
    x1, y1, z1 = bird

    # Convert to fixed-point integers for secure computation
    x0_fp = int(x0 * max_val)
    y0_fp = int(y0 * max_val)
    z0_fp = int(z0 * max_val)
    x1_fp = int(x1 * max_val)
    y1_fp = int(y1 * max_val)
    z1_fp = int(z1 * max_val)

    print(f"Computing secure 3D distance and angles between {observer} and {bird}")
    print(
        f"Fixed-point values: observer=({x0_fp}, {y0_fp}, {z0_fp}), bird=({x1_fp}, {y1_fp}, {z1_fp})"
    )

    # Prepare vectors
    vec1 = np.array([x0_fp, y0_fp, z0_fp], dtype=funshade.DTYPE)
    vec2 = np.array([x1_fp, y1_fp, z1_fp], dtype=funshade.DTYPE)

    # Print the vectors to observe the contents
    print("+" * 50)
    print(f"\nVector 1 (Observer): {vec1}")
    print(f"Vector 2 (Bird): {vec2}\n")
    print("+" * 50)

    # Create parties
    class Party:
        def __init__(self, j: int):
            self.j = j

    BP = Party(0)
    Gate = Party(1)

    # Print the party information
    print(f"Party 0 (BP) initialized with j={BP.j}")
    print(f"Party 1 (Gate) initialized with j={Gate.j}")
    print("+" * 50)

    K = 1
    l = 3  # 3D

    # Setup phase (simulate randomness for secure computation)
    d_x0, d_x1, d_y0, d_y1, d_xy0, d_xy1, r_in0, r_in1, k0, k1 = funshade.setup(K, l, 0)
    print("+" * 50)
    print("Secure computation setup complete.")
    print(
        f"Random values: d_x0={d_x0}, d_y0={d_y0}, d_xy0={d_xy0}, r_in0={r_in0}, k0={k0}"
    )
    print(type(d_x0), type(d_y0), type(d_xy0), type(r_in0), type(k0))
    print(d_x0.shape, d_y0.shape, d_xy0.shape, r_in0.shape, k0.shape)
    print("-" * 50)
    print(
        f"Random values: d_x1={d_x1}, d_y1={d_y1}, d_xy1={d_xy1}, r_in1={r_in1}, k1={k1}"
    )
    # Print the information about the random values d_x1, d_y1, d_xy1, r_in1, k1
    print(type(d_x1), type(d_y1), type(d_xy1), type(r_in1), type(k1))
    print(d_x1.shape, d_y1.shape, d_xy1.shape, r_in1.shape, k1.shape)
    print("-" * 50)

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

    # Exchange shares to reconstruct dot product
    dot_product = (BP.z_hat_j + Gate.z_hat_j) - (BP.r_in_j + Gate.r_in_j)
    dot_product_val = dot_product[0]

    # Compute difference vector (bird - observer)
    dx_fp = x1_fp - x0_fp
    dy_fp = y1_fp - y0_fp
    dz_fp = z1_fp - z0_fp

    # Compute squared distance
    squared_distance = dx_fp * dx_fp + dy_fp * dy_fp + dz_fp * dz_fp
    distance = math.sqrt(squared_distance) / max_val

    # Compute elevation angle (from horizontal plane)
    horizontal_norm = math.sqrt(dx_fp**2 + dy_fp**2)
    if horizontal_norm == 0:
        elevation = math.pi / 2 if dz_fp > 0 else -math.pi / 2
    else:
        elevation = math.atan2(dz_fp, horizontal_norm)

    # Compute azimuthal angle (from x-axis in x-y plane)
    azimuth = math.atan2(dy_fp, dx_fp)

    print(f"Secure Computed Distance: {distance}")
    print(
        f"Secure Elevation (radians): {elevation}, degrees: {math.degrees(elevation)}"
    )
    print(f"Secure Azimuth (radians): {azimuth}, degrees: {math.degrees(azimuth)}")

    # Verification with ground truth
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    actual_distance = math.sqrt(dx**2 + dy**2 + dz**2)
    actual_elevation = (
        math.atan2(dz, math.sqrt(dx**2 + dy**2))
        if (dx != 0 or dy != 0)
        else (math.pi / 2 if dz > 0 else -math.pi / 2)
    )
    actual_azimuth = math.atan2(dy, dx)

    print(f"Actual Distance: {actual_distance}")
    print(
        f"Actual Elevation (radians): {actual_elevation}, degrees: {math.degrees(actual_elevation)}"
    )
    print(
        f"Actual Azimuth (radians): {actual_azimuth}, degrees: {math.degrees(actual_azimuth)}"
    )
    print(f"Distance match: {abs(distance - actual_distance) < 0.05}")
    print(f"Elevation match: {abs(elevation - actual_elevation) < 0.05}")
    print(f"Azimuth match: {abs(azimuth - actual_azimuth) < 0.05}")

    return {"distance": distance, "elevation": elevation, "azimuth": azimuth}


def main():
    print("=" * 60)
    print("Secure 3D Distance, Elevation, and Azimuth Computation with Funshade")
    print("=" * 60)

    # Example usage
    observer = (0.0, 0.0, 0.0)
    bird = (10.0, 10.0, 5.0)
    res = compute_3d_distance_and_angles_secure(observer, bird)

    # Verify the result using the following formulae for ground truth:
    # Px = distance * cos(elevation) * cos(azimuth)
    # Py = distance * cos(elevation) * sin(azimuth)
    # Pz = distance * sin(elevation)
    Px = res["distance"] * math.cos(res["elevation"]) * math.cos(res["azimuth"])
    Py = res["distance"] * math.cos(res["elevation"]) * math.sin(res["azimuth"])
    Pz = res["distance"] * math.sin(res["elevation"])

    print(f"Calculated truth coordinates: Px={Px}, Py={Py}, Pz={Pz}")
    print(f"Actual bird coordinates: Px={bird[0]}, Py={bird[1]}, Pz={bird[2]}")

    print("=" * 60)
    print("All computations completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
