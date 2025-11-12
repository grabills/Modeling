import Pkg; Pkg.add("GLMakie")
using GLMakie

# 1. Define the gyroid implicit function
gyroid(x, y, z) = sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)

# 2. Set up the grid for the unit cell
n = 100 # Resolution of the grid
x = LinRange(-pi, pi, n)
y = LinRange(-pi, pi, n)
z = LinRange(-pi, pi, n)

# 3. Evaluate the gyroid function on the 3D grid
vals = Float32[gyroid(i, j, k) for i in x, j in y, k in z]

# 4. Create the 3D plot
fig = Figure(size = (800, 800))

# --- CORRECTION IS HERE ---
ax = Axis3(fig[1, 1],
    title = "Gyroid Surface within its Unit Cell",
    aspect = :data,
    # Corrected keyword: `panel` instead of `pane`
    xypanelvisible = true,
    yzpanelvisible = true,
    xzpanelvisible = true,
    xticksvisible = false, yticksvisible = false, zticksvisible = false,
    xlabel = "X", ylabel = "Y", zlabel = "Z"
)
# --- END OF CORRECTION ---

# Lock the view to the exact boundaries of the unit cell
limits!(ax, -pi, pi, -pi, pi, -pi, pi)

# Define the thickness
thickness = 1.0

# Plot the surfaces that bound the thick volume
contour!(ax, (-pi, pi), (-pi, pi), (-pi, pi), vals, 
         levels=[-thickness, thickness], # This now defines the boundaries
         colormap=:viridis)

# Display the interactive plot
display(fig)

println("Press Enter to close the plot.")
readline()
