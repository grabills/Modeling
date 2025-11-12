import Pkg; Pkg.add("GLMakie")
using GLMakie
using LinearAlgebra # For pi

# --- PARAMETRIC "FLÄCHENSTÜCK" FUNCTIONS ---
# (s_param, t_param, f_func, X_com_complex, Y_com_complex, Z_com_complex)
# These are identical to before, as the base formula is the same.

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

# --- G-SURFACE (GYROID) CONSTRUCTION ---

# 1. Set up the grid for the base parameters `v` and `w`
# Per the text for the polynomial approximation, we use the 4-sided square
n = 100 # Resolution
v_range = LinRange(-1, 1, n)
w_range = LinRange(-1, 1, n)

# Get parameters from text (Table II for G-Surface)
G_SCALE = 0.376472
G_THETA = 0.66349 # The association angle in radians

# Complex rotation factor: e^(i*θ)
rot_factor = exp(im * G_THETA)

# 2. Evaluate the functions for the G-Surface patch
println("Evaluating $(n^2) points for the G-Surface 'Flächenstück'...")

X_grid = fill(NaN, n, n)
Y_grid = fill(NaN, n, n)
Z_grid = fill(NaN, n, n)

for (i, v) in enumerate(v_range), (j, w) in enumerate(w_range)
    
    # NOTE: We are plotting the full 4-sided square: -1 ≤ v, w ≤ 1
    # The (g, h) boundary (Eq. 9) is NOT for this approximation method.
    
    # Calculate complex coordinates
    xc = X_com_complex(v, w)
    yc = Y_com_complex(v, w)
    zc = Z_com_complex(xc, yc)
        
    # Apply Eq. 35: (X,Y,Z) = G_SCALE * Real( e^(i*θ) * (X_c, Y_c, Z_c) )
    X_grid[i, j] = G_SCALE * real(rot_factor * xc)
    Y_grid[i, j] = G_SCALE * real(rot_factor * yc)
    Z_grid[i, j] = G_SCALE * real(rot_factor * zc)
end
println("Evaluation complete.")


# 3. Create the 3D plot
fig = Figure(size = (800, 800))

ax = Axis3(fig[1, 1],
    title = "G-Surface (Gyroid) 'Flächenstück' (4-Sided)",
    aspect = :data,
    xypanelvisible = true,
    yzpanelvisible = true,
    xzpanelvisible = true
)

# 4. Plot the single G-Surface patch
surface!(ax, 
    X_grid, Y_grid, Z_grid, 
    colormap=:viridis, 
    shading=true
)

display(fig)

println("Press Enter to close the plot.")
readline()
