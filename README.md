# Schwarz P Surface Unit Cell Generator

## Overview

This Julia project generates a high-accuracy 3D mesh of the Schwarz P minimal surface unit cell. Unlike standard implementations that use implicit trigonometric level sets (e.g., `cos(x) + cos(y) + cos(z) = 0`), this implementation constructs the surface using a parametric polynomial approximation.

The construction process involves two main phases:
1.  **Base Patch Generation:** Computing a fundamental surface patch ("Flächenstück") using a complex polynomial parameterization (Equations 4.32, 4.33, and 4.34 from the reference literature).
2.  **Unit Cell Assembly:** Applying a specific sequence of affine transformations—scaling, rotation, translation, and inversion—to "quilt" copies of the base patch into a complete, water-tight unit cell.

## Dependencies

This project requires the following Julia packages for geometry handling, linear algebra operations, file input/output, and visualization:

* `GeometryBasics.jl`: For mesh data structures (vertices and faces).
* `CoordinateTransformations.jl`: For applying affine transformations.
* `Rotations.jl`: For handling 3D rotation matrices.
* `FileIO.jl`: For file system interactions.
* `MeshIO.jl`: For saving the output to mesh formats (STL).
* `GLMakie.jl`: For interactive 3D visualization.
* `LinearAlgebra`: (Standard Library) For matrix operations.

## Installation and Setup

1.  **Install Julia**: Ensure you have a recent version of Julia installed on your system.
2.  **Install Packages**: Open the Julia REPL (command line) and enter package mode by pressing `]`. Run the following command to install the required dependencies:

```julia
pkg> add GeometryBasics CoordinateTransformations Rotations FileIO MeshIO GLMakie
