import Pkg; Pkg.add("GLMakie")
using GLMakie
using LinearAlgebra # For pi
using GeometryBasics 

# --- CORE "FLÄCHENSTÜCK" FORMULAS ---
# These are the underlying math functions, independent of parameterization

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

function Z_com_complex(X_c::Complex, Y_c::Complex)
    term1 = 0.5 * (X_c^2 
- Y_c^2)
    # --- THIS IS THE FIX ---
    # The coefficient in your file  was 0.20659, which causes intersections.
    term2 = 0.020659 * (X_c^4 - Y_c^4) 
    term3 = -0.01268 * (X_c^6 - Y_c^6)
    return term1 + term2 + term3
end

# This function defines the lower 't' boundary from p. 91
function t_min_func(s)
    s2 = s^2
    s4 = s^4
    s6 = s^6
    s8 = s^8
    # From line (4.35) on page 91
    # Combining the two s^8 terms
    t = -0.121906*s2 + 0.052357*s4 + 0.101371*s6 - 0.03182274*s8
    return t
end


# --- G-SURFACE (GYROID) CONSTRUCTION ---
println("Generating vertices for 6 patches based on p. 91 'basic subsection'...")

# 1. Set up the grid for the *basic subsection* (p. 91)
n_s = 50 # Resolution along 's'
n_t = 50 # Resolution along 't'

s_range = LinRange(-1, 1, n_s)
t_map_range = LinRange(0, 1, n_t) # We use a "mapping" parameter from 0 to 1

# Get parameters from text (Table II for G-Surface)
G_SCALE = 0.376472
G_THETA = 0.66349 # The association angle in radians

# Complex rotation factor: e^(i*θ)
rot_factor = exp(im * G_THETA)

# --- Step 1: Generate Vertices for the *Base Patch* ---
all_vertices = Point3f[]
base_patch_vertices = Point3f[]

for (i, s) in enumerate(s_range), (j, t_map) in enumerate(t_map_range)
    
    # Map 't_map' from [0, 1] to the curvilinear domain [t_min, t_max]
    # These are the inequalities from page 91
    t_max = 1 - abs(s)
    t_min = t_min_func(s)
    
    # This maps our clean grid to the curved patch
    t = (1 - t_map) * t_min + t_map * t_max

    # --- Core Calculation ---
    f1 = f_func(s, t)
    f2 = f_func(-t, s)
    
    xc = Complex(s, f1)
    yc = Complex(t, -f2)
    zc = Z_com_complex(xc, yc)
    
    # Apply Eq. 35: (X,Y,Z) = G_SCALE * Real( e^(i*θ) * (X_c, Y_c, Z_c) )
    X = G_SCALE * real(rot_factor * xc)
    Y = G_SCALE * real(rot_factor * yc)
    Z = G_SCALE * real(rot_factor * zc)
    
    # We do NOT apply the translation. Symmetries must be origin-centered.
    push!(base_patch_vertices, Point3f(X, Y, Z))
end

# --- Step 1b: Apply Symmetries to Vertices ---
append!(all_vertices, base_patch_vertices)
N_patch = length(base_patch_vertices) # Number of vertices per patch (n_s * n_t)

# P2 Vertices: (y, z, x) rotation
p2_vertices = [Point3f(p[2], p[3], p[1]) for p in base_patch_vertices]
append!(all_vertices, p2_vertices)

# P3 Vertices: (z, x, y) rotation
p3_vertices = [Point3f(p[3], p[1], p[2]) for p in base_patch_vertices]
append!(all_vertices, p3_vertices)

# P4, P5, P6 Vertices: Inversion
p123_vertices = all_vertices
p456_vertices = [-p for p in p123_vertices]
append!(all_vertices, p456_vertices)

println("Total vertices generated: $(length(all_vertices))")

# --- Step 2: Generate All Faces ---
println("Generating faces for 6 patches...")

all_faces = TriangleFace{Int}[] 
base_faces = TriangleFace{Int}[] 

# Helper function to get 1D index from 2D (i, j) grid indices
idx(i, j, n) = (i - 1) * n + j

# Loop over the quads of our (s, t_map) grid
for i in 1:(n_s-1), j in 1:(n_t-1)
    idx1 = idx(i, j, n_t)   # Bottom-left
    idx2 = idx(i+1, j, n_t) # Bottom-right
    idx3 = idx(i, j+1, n_t)   # Top-left
    idx4 = idx(i+1, j+1, n_t) # Top-right

    # Split the quad into two triangles
    push!(base_faces, TriangleFace(idx1, idx2, idx4)) # Triangle 1
    push!(base_faces, TriangleFace(idx1, idx4, idx3)) # Triangle 2
end

# Now, copy and offset the face indices for the other 5 patches
append!(all_faces, base_faces) # F1 (Faces for P1)
append!(all_faces, [f .+ N_patch for f in base_faces])       # F2 (Faces for P2)
append!(all_faces, [f .+ (2 * N_patch) for f in base_faces]) # F3 (Faces for P3)
append!(all_faces, [f .+ (3 * N_patch) for f in base_faces]) # F4 (Faces for P4)
append!(all_faces, [f .+ (4 * N_patch) for f in base_faces]) # F5 (Faces for P5)
append!(all_faces, [f .+ (5 * N_patch) for f in base_faces]) # F6 (Faces for P6)

println("Total faces generated: $(length(all_faces))")

# --- END: NEW MESHING ALGORITHM ---

# --- Combine into a single Mesh object ---
println("Combining vertices and faces into a single mesh...")
gyroid_mesh = GeometryBasics.Mesh(all_vertices, all_faces)
println("Mesh object created successfully.")


# 3. Create the 3D plot
fig = Figure(size = (800, 800))

ax = Axis3(fig[1, 1],
    title = "G-Surface 6-Patch Mesh (from p. 91)",
    aspect = :data,
    xypanelvisible = true,
    yzpanelvisible = true,
    xzpanelvisible = true
)

# 4. Plot the single, unified wireframe
wireframe!(ax, 
    gyroid_mesh,
    color=:black, 
    linewidth=0.5
)

display(fig)

println("Press Enter to close the plot.")
readline()
