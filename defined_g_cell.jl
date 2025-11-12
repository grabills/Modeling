import Pkg; Pkg.add("GLMakie")
using GLMakie

# --- MODIFIED SECTION ---

# We can use a slightly higher limit, since the
# coefficients will decay quickly.
const N_LIMIT = 4

# This is the new coefficient function.
# F_hkl is now a physically-motivated decaying function.
function F_coeff(h, k, l)
    if h == 0 && k == 0 && l == 0
        return 0.0 # This term is skipped anyway
    end
    
    # Set amplitude to be 1 / (h^2 + k^2 + l^2)
    # This penalizes high-frequency terms and is physically-motivated.
    return 1.0 / (h^2 + k^2 + l^2)
end

# --- END OF MODIFIED SECTION ---


# 1. Define the Fourier series function $\rho(x, y, z)$
# (This function is identical to before)
function rho(x, y, z)
    total_sum = 0.0
    
    px = 2π * x
    py = 2π * y
    pz = 2π * z

    for h in 0:N_LIMIT, k in 0:N_LIMIT, l in 0:N_LIMIT
        if h == 0 && k == 0 && l == 0
            continue # Skip (0,0,0) term, F_coeff is 0 anyway
        end

        f_hkl = F_coeff(h, k, l)

        # Check the parity conditions from your formula
        h_plus_k_even = (h + k) % 2 == 0
        k_plus_l_even = (k + l) % 2 == 0

        if h_plus_k_even && k_plus_l_even
            # Term 1: F(h+k=2n, k+l=2n)
            total_sum += f_hkl * cos(px * h) * cos(py * k) * cos(pz * l)
        
        elseif h_plus_k_even && !k_plus_l_even
            # Term 2: F(h+k=2n, k+l=2n+1)
            total_sum -= f_hkl * sin(px * h) * sin(py * k) * cos(pz * l)

        elseif !h_plus_k_even && k_plus_l_even
            # Term 3: F(h+k=2n+1, k+l=2n)
            total_sum -= f_hkl * cos(px * h) * sin(py * k) * sin(pz * l)

        elseif !h_plus_k_even && !k_plus_l_even
            # Term 4: F(h+k=2n+1, k+l=2n+1)
            total_sum -= f_hkl * sin(px * h) * cos(py * k) * sin(pz * l)
        end
    end
    return total_sum
end

# 2. Set up the grid for the unit cell
n = 50 # Resolution
x = LinRange(0, 1, n)
y = LinRange(0, 1, n)
z = LinRange(0, 1, n)

# 3. Evaluate the function on the 3D grid
println("Evaluating $(n^3) points with N=$(N_LIMIT) sums (decaying coeffs)...")
vals = Float32[rho(i, j, k) for i in x, j in y, k in z]
println("Evaluation complete.")

# 4. Create the 3D plot
fig = Figure(size = (800, 800))

ax = Axis3(fig[1, 1],
    title = "Schwarz D-Surface (Fourier N=$(N_LIMIT), Decaying Coeffs)",
    aspect = :data,
    xypanelvisible = true,
    yzpanelvisible = true,
    xzpanelvisible = true
)

# Lock the view to the (0, 1) unit cell
limits!(ax, 0, 1, 0, 1, 0, 1)

# Plot the isosurface where rho = 0
# You may need to adjust this level slightly (e.g., to 0.1 or -0.1)
# if 0.0 doesn't show a surface, depending on the constant
# offset (the k=0,h=0,l=0 term we skipped).
contour!(ax, (0, 1), (0, 1), (0, 1), vals,
         levels=[0.05], # Starting with a small offset level
         colormap=:viridis)

display(fig)

println("Press Enter to close the plot.")
readline()
