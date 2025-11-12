using Makie, GLMakie

# 1. Define the Schwarz D-surface implicit function
d_surface(x, y, z) = cos(x) * cos(y) * cos(z) - sin(x) * sin(y) * sin(z)

# 2. Define a clipping function to isolate a single unit
# This function will only show the surface inside a sphere of a given radius.
function clipped_surface(x, y, z, radius)
    # Check if the point (x,y,z) is inside the sphere
    if x^2 + y^2 + z^2 < radius^2
        return d_surface(x, y, z) # If inside, return the surface value
    else
        return Inf32 # If outside, return infinity to hide it
    end
end

# 3. Set up the grid for the unit cell
n = 100 # Resolution of the grid
x = LinRange(-pi, pi, n)
y = LinRange(-pi, pi, n)
z = LinRange(-pi, pi, n)

# 4. Evaluate the CLIPPED D-surface function on the 3D grid
# We set a clipping radius to carve out the central shape.
clip_radius = pi / sqrt(2)
vals = Float32[clipped_surface(i, j, k, clip_radius) for i in x, j in y, k in z]

# 5. Create the 3D plot
fig = Figure(size = (800, 800))
ax = Axis3(fig[1, 1],
    title = "Single D-Surface Fundamental Unit",
    aspect = :data,
    xypanelvisible = true,
    yzpanelvisible = true,
    xzpanelvisible = true,
    xticksvisible = false, yticksvisible = false, zticksvisible = false,
    xlabel = "X", ylabel = "Y", zlabel = "Z"
)

# Lock the view to the exact boundaries of the unit cell
limits!(ax, -pi, pi, -pi, pi, -pi, pi)

# Use the `volume` command to create the isosurface
volume!(ax, (-pi, pi), (-pi, pi), (-pi, pi), vals, algorithm = :iso, isovalue = 0.0, colormap=:viridis)

# Display the interactive plot
display(fig)

println("Press Enter to close the plot.")
readline()
