import GLMakie, Makie
using Makie, GLMakie

# 1. Define the Schwarz P-surface implicit function
p_surface(x, y, z) = cos(x) + cos(y) + cos(z)

# 2. Set up the grid for the unit cell
# --- DOMAIN CORRECTED HERE ---
# The domain is expanded to [-π, π] to capture the full surface.
n = 500 # Resolution of the grid
x = LinRange(-pi, pi, n)
y = LinRange(-pi, pi, n)
z = LinRange(-pi, pi, n)
# --- END CORRECTION ---

# 3. Evaluate the P-surface function on the 3D grid
vals = Float64[p_surface(i, j, k) for i in x, j in y, k in z]

# 4. Create the 3D plot
fig = Figure(size = (800, 800))

ax = Axis3(fig[1, 1],
    title = "Schwarz P-Surface",
    aspect = :data,
    xypanelvisible = true,
    yzpanelvisible = true,
    xzpanelvisible = true,
    xticksvisible = false, yticksvisible = false, zticksvisible = false,
    xlabel = "X", ylabel = "Y", zlabel = "Z"
)

# --- DOMAIN CORRECTED HERE ---
# Lock the view to the new, larger boundaries.
limits!(ax, -pi, pi, -pi, pi, -pi, pi)
# --- END CORRECTION ---

thickness = 0.1

# Plot the surfaces that bound the thick volume
contour!(ax, (-pi, pi), (-pi, pi), (-pi, pi), vals, 
         levels=[-thickness, thickness], # This now defines the boundaries
         colormap=:viridis)


# Display the interactive plot
display(fig)

println("Press Enter to close the plot.")
readline()
