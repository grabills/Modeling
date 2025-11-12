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

# --- P-SURFACE (PRIMITIVE) UNIT CELL CONSTRUCTION ---
# This version IGNORES the flawed "move" and "doubling" steps
# from the text and correctly builds the central node.

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
# Step 1: "Begin with the top quadrant... (0 ≤ v, w ≤ 1)"
v_range = LinRange(0, 1, n)
w_range = LinRange(0, 1, n)

# 3. Get P-Cell parameters
P_SCALE = 0.463677 #

# 4. Define the BASE SHAPE from Step 1 and 2 (origin-centered)
#    This defines the 3-patch cluster that forms ONE "arm"
T_base_shape = P_SCALE * rot_z(3*pi/4) # Step 1 rot
T_sym2 = rot_z(pi/2) * rot_y(-pi/2)     # Step 2 rot
T_sym3 = rot_y(pi/2) * rot_z(-pi/2)     # Step 2 rot

# The 3 shapes that make one arm
Arm_Shapes = [
    T_base_shape,
    T_sym2 * T_base_shape,
    T_sym3 * T_base_shape
]

# 5. Define the 6 rotations to build the full node
#    This replaces the flawed steps 4-7.
#    We need 6 arms, pointing along +x, -x, +y, -y, +z, -z.
#    The base arm (defined above) points roughly towards +z.
Arm_Rotations = [
    I(3),                     # +z arm (base)
    rot_x(pi),                # -z arm
    rot_y(pi/2),              # +x arm
    rot_y(-pi/2),             # -x arm
    rot_x(-pi/2),             # +y arm
    rot_x(pi/2)               # -y arm
]

# 6. Prepare grids for all 18 final patches (3 shapes * 6 rotations)
patch_grids = [ (fill(NaN, n, n), fill(NaN, n, n), fill(NaN, n, n)) for _ in 1:18 ]

println("Evaluating $(n^2) points for the P-Surface Central Node (18 patches)...")

# 7. Evaluate the functions and apply all transformations
for (i, v) in enumerate(v_range), (j, w) in enumerate(w_range)
    
    # Calculate complex coordinates
    xc = X_com_complex(v, w)
    yc = Y_com_complex(v, w)
    zc = Z_com_complex(xc, yc)
        
    # Base 3D point from real parts
    p_base = [real(xc), real(yc), real(zc)]

    # --- *** NEW FIXED LOGIC (v6) *** ---
    
    k = 1 # Patch counter
    for T_arm_rot in Arm_Rotations # Loop over 6 arm rotations
        for T_shape in Arm_Shapes # Loop over 3 base shapes
            
            # Apply base shape rot, then arm rot
            p_final = T_arm_rot * T_shape * p_base
            
            # Store result
            patch_grids[k][1][i, j] = p_final[1] # X
            patch_grids[k][2][i, j] = p_final[2] # Y
            patch_grids[k][3][i, j] = p_final[3] # Z
            k += 1
        end
    end
end
println("Evaluation complete.")


# 8. Create the 3D plot
fig = Figure(size = (800, 800))
ax = Axis3(fig[1, 1],
    title = "P-Surface (Primitive) Unit Cell (Corrected)",
    aspect = :data,
    xypanelvisible = true,
    yzpanelvisible = true,
    xzpanelvisible = true
)

# 9. Plot all 18 patches
for (X, Y, Z) in patch_grids
    surface!(ax, 
        X, Y, Z, 
        colormap=:viridis, 
        shading=true
    )
end

# Set a good starting camera angle (matches target image)
ax.azimuth = 2.75
ax.elevation = 0.5
ax.perspectiveness = 0.5

# Set axis limits to match target image
ax.limits = (-0.6, 0.6, -0.6, 0.6, -0.6, 0.6)

display(fig)

println("Press Enter to close the plot.")
readline()
