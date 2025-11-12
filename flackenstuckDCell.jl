import Pkg; Pkg.add("GLMakie")
using GLMakie
using LinearAlgebra # For pi and matrix operations

# --- PARAMETRIC "FLÄCHENSTÜCK" FUNCTIONS ---
# These functions define the base surface patch and are unchanged.
# 
s_param(v, w) = 0.298 * (v + w)
t_param(v, w) = 0.298 * (-v + w)

function f_func(s, t)
    term1 = -1.00102 * t
    term2 = 2.00914 * s^2 * t
    term3 = -0.69992 * s^4 * t
    term4 = -5.19821 * s^6 * t
    term5 = -0.62234 * t^3
    term6 = -1.18001 * s^2 * t^3
    term7 = 28.34854 * s^4 * t^3
    term8 = -1.21488 * t^5
    term9 = -16.83032 * s^2 * t^5
    term10 = 2.14239 * t^7
    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10
end

function X_com_complex(v, w)
    s = s_param(v, w)
    t = t_param(v, w)
    f = f_func(s, t)
    return Complex(s, f)
end

function Y_com_complex(v, w)
    s = s_param(v, w)
    t = t_param(v, w)
    f = f_func(-t, s)
    return Complex(t, -f)
end

function Z_com_complex(X_c::Complex, Y_c::Complex)
    term1 = 0.5 * (X_c^2 - Y_c^2)
    term2 = 0.20659 * (X_c^4 - Y_c^4)
    term3 = -0.01268 * (X_c^6 - Y_c^6)
    return term1 + term2 + term3
end

# --- D-SURFACE (DIAMOND) UNIT CELL CONSTRUCTION ---
# Based on instructions from the provided image.

# 1. Helper functions for 3D rotations
function rot_x(θ)
    return [1 0 0; 0 cos(θ) -sin(θ); 0 sin(θ) cos(θ)]
end
function rot_y(θ)
    return [cos(θ) 0 sin(θ); 0 1 0; -sin(θ) 0 cos(θ)]
end
function rot_z(θ)
    return [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1]
end

# 2. Set up the grid for the base parameters `v` and `w`
n = 100 # Resolution
v_range = LinRange(-1, 1, n) # As per instruction " -1 ≤ v, w ≤ 1" 
w_range = LinRange(-1, 1, n) # 

# 3. Get D-Cell parameters from the instructions text
D_SCALE = 0.593207
D_TRANSLATE = [1/4, -1/4, -1/4]

# 4. Define all transformation matrices from the instructions
# Step 1: Base patch transformation
T1_rot = rot_z(3*pi/4) # R_z + 3π/4
T1_scale = D_SCALE
T1_trans = D_TRANSLATE

# Step 2: 3-fold symmetry rotations (applied to P1)
T2_rot = rot_z(pi/2) * rot_y(-pi/2) # R_z+π/2 * R_y-π/2
T3_rot = rot_y(pi/2) * rot_z(-pi/2) # R_y+π/2 * R_z-π/2

# Step 3: Inversion (applied to P1, P2, P3)
T_inv = -I(3) # I_x * I_y * I_z is point inversion (-x, -y, -z)

# Step 4: Final rotation (applied to all 6 patches)
T_final_rot = rot_x(pi/2) # R_x+π/2

# 5. Prepare grids for all 6 final patches
# We will have 6 (X, Y, Z) grids, one for each patch
patch_grids = [ (fill(NaN, n, n), fill(NaN, n, n), fill(NaN, n, n)) for _ in 1:6 ]

println("Evaluating $(n^2) points for the D-Surface Unit Cell...")

# 6. Evaluate the functions and apply all transformations
for (i, v) in enumerate(v_range), (j, w) in enumerate(w_range)
    
    # Calculate complex coordinates 
    xc = X_com_complex(v, w)
    yc = Y_com_complex(v, w)
    zc = Z_com_complex(xc, yc)
        
    # Base 3D point is the real part of the complex coordinates
    p_base = [real(xc), real(yc), real(zc)]

    # --- Apply symmetry operations as per the image ---

    # Step 1: Create Patch 1
    p1 = (T1_scale * T1_rot * p_base) + T1_trans

    # Step 2: Create Patches 2 & 3 from P1
    p2 = T2_rot * p1
    p3 = T3_rot * p1

    # Step 3: Create Patches 4, 5, 6 by inverting P1, P2, P3
    p4 = T_inv * p1
    p5 = T_inv * p2
    p6 = T_inv * p3

    # Step 4: Apply final rotation to all 6 patches
    patches_raw = [p1, p2, p3, p4, p5, p6]
    patches_final = [T_final_rot * p for p in patches_raw]

    # Store results in their respective grids
    for k in 1:6
        patch_grids[k][1][i, j] = patches_final[k][1] # X
        patch_grids[k][2][i, j] = patches_final[k][2] # Y
        patch_grids[k][3][i, j] = patches_final[k][3] # Z
    end
end
println("Evaluation complete.")


# 7. Create the 3D plot
fig = Figure(size = (800, 800))
ax = Axis3(fig[1, 1],
    title = "D-Surface (Diamond) Unit Cell",
    aspect = :data,
    xypanelvisible = true,
    yzpanelvisible = true,
    xzpanelvisible = true
)

# 8. Plot all 6 patches
for (X, Y, Z) in patch_grids
    surface!(ax, 
        X, Y, Z, 
        colormap=:viridis, 
        shading=true
    )
end

# Set a good starting camera angle
ax.azimuth = 1.95
ax.elevation = 0.5
ax.perspectiveness = 0.5

display(fig)

println("Press Enter to close the plot.")
readline()
