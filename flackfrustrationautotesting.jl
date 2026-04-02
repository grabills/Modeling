using GLMakie
using LinearAlgebra
using Statistics
using CUDA

# --- 1. NODAL APPROXIMATIONS (GPU COMPATIBLE) ---
# 1 = Gyroid, 2 = Primitive, 3 = Diamond
const SURF_CODES = Dict("Gyroid (G)" => 1, "Primitive (P)" => 2, "Diamond (D)" => 3)

function eval_surface_gpu(type_code::Int, x::Float32, y::Float32, z::Float32)
    if type_code == 1
        return sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x)
    elseif type_code == 2
        return cos(x) + cos(y) + cos(z)
    elseif type_code == 3
        return cos(x)*cos(y)*cos(z) - sin(x)*sin(y)*sin(z)
    end
    return 0.0f0
end

# --- 2. DIFFERENTIAL GEOMETRY (GPU COMPATIBLE) ---
function calc_curvature_implicit_gpu(type_code::Int, x::Float32, y::Float32, z::Float32, delta::Float32=2f-2)
    Fx = (eval_surface_gpu(type_code, x+delta, y, z) - eval_surface_gpu(type_code, x-delta, y, z)) / (2f0*delta)
    Fy = (eval_surface_gpu(type_code, x, y+delta, z) - eval_surface_gpu(type_code, x, y-delta, z)) / (2f0*delta)
    Fz = (eval_surface_gpu(type_code, x, y, z+delta) - eval_surface_gpu(type_code, x, y, z-delta)) / (2f0*delta)
    
    grad_mag2 = Fx^2 + Fy^2 + Fz^2
    
    K_out = 0.0f0
    if grad_mag2 >= 1f-6
        Fxx = (eval_surface_gpu(type_code, x+delta, y, z) - 2f0*eval_surface_gpu(type_code, x, y, z) + eval_surface_gpu(type_code, x-delta, y, z)) / (delta^2)
        Fyy = (eval_surface_gpu(type_code, x, y+delta, z) - 2f0*eval_surface_gpu(type_code, x, y, z) + eval_surface_gpu(type_code, x, y-delta, z)) / (delta^2)
        Fzz = (eval_surface_gpu(type_code, x, y, z+delta) - 2f0*eval_surface_gpu(type_code, x, y, z) + eval_surface_gpu(type_code, x, y, z-delta)) / (delta^2)
        
        Fxy = (eval_surface_gpu(type_code, x+delta, y+delta, z) - eval_surface_gpu(type_code, x-delta, y+delta, z) - 
               eval_surface_gpu(type_code, x+delta, y-delta, z) + eval_surface_gpu(type_code, x-delta, y-delta, z)) / (4f0*delta^2)
        Fxz = (eval_surface_gpu(type_code, x+delta, y, z+delta) - eval_surface_gpu(type_code, x-delta, y, z+delta) - 
               eval_surface_gpu(type_code, x+delta, y, z-delta) + eval_surface_gpu(type_code, x-delta, y, z-delta)) / (4f0*delta^2)
        Fyz = (eval_surface_gpu(type_code, x, y+delta, z+delta) - eval_surface_gpu(type_code, x, y-delta, z+delta) - 
               eval_surface_gpu(type_code, x, y+delta, z-delta) + eval_surface_gpu(type_code, x, y-delta, z-delta)) / (4f0*delta^2)

        num = Fx^2*(Fyy*Fzz - Fyz^2) + Fy^2*(Fxx*Fzz - Fxz^2) + Fz^2*(Fxx*Fyy - Fxy^2) +
              2f0*Fx*Fy*(Fxz*Fyz - Fxy*Fzz) + 2f0*Fy*Fz*(Fxy*Fxz - Fyz*Fxx) + 2f0*Fx*Fz*(Fxy*Fyz - Fxz*Fyy)
        
        K_out = num / (grad_mag2^2)
    end
    return K_out
end

# --- 3. BIOPHYSICS & THERMODYNAMICS ---
const PACKING_VAR = Dict(
    "Lamellar"    => 0.0,  
    "Gyroid (G)"  => 1.0,  
    "Diamond (D)" => 1.15, 
    "Primitive (P)"=> 1.65 
)

function calc_area_frustration(K, l_lipid, A_opt, k_a=1.0)
    local_area_factor = (1.0 + (l_lipid^2) * K)
    return 0.5 * k_a * ((local_area_factor - A_opt)^2)
end

function calc_total_energy(surf_type, k_array, l_lipid, A_opt, k_p)
    if isempty(k_array) && surf_type != "Lamellar" return 0.0 end
    
    if surf_type == "Lamellar"
        mean_area_E = 0.5 * ((1.0 - A_opt)^2) 
    else
        total_a_e = 0.0
        for K in k_array
            total_a_e += calc_area_frustration(K, l_lipid, A_opt)
        end
        mean_area_E = total_a_e / length(k_array)
    end
    
    packing_E = 0.5 * k_p * PACKING_VAR[surf_type] * (l_lipid^2)
    return mean_area_E + packing_E
end

# --- 4. CUDA ACCELERATED GEOMETRY CACHE ---
const CACHE_PTS = Dict{String, Vector{Point3f}}()
const CACHE_K   = Dict{String, Vector{Float32}}()

function build_geometry_cache!(b_mult)
    println("Uploading domain to NVIDIA GPU for processing...")
    empty!(CACHE_PTS); empty!(CACHE_K)
    
    res = 55
    bound = Float32(b_mult * π)
    r = Float32.(range(-bound, stop=bound, length=res))
    epsilon = 0.12f0
    
    grid_x = Float32[x for x in r, y in r, z in r][:]
    grid_y = Float32[y for x in r, y in r, z in r][:]
    grid_z = Float32[z for x in r, y in r, z in r][:]
    
    cu_X = CuArray(grid_x)
    cu_Y = CuArray(grid_y)
    cu_Z = CuArray(grid_z)
    
    for surf in ["Gyroid (G)", "Primitive (P)", "Diamond (D)"]
        code = SURF_CODES[surf]
        cu_F = eval_surface_gpu.(code, cu_X, cu_Y, cu_Z)
        cu_K = calc_curvature_implicit_gpu.(code, cu_X, cu_Y, cu_Z)
        
        F_host = Array(cu_F)
        K_host = Array(cu_K)
        
        pts = Point3f[]
        k_vals = Float32[]
        for i in 1:length(F_host)
            if abs(F_host[i]) < epsilon
                push!(pts, Point3f(grid_x[i], grid_y[i], grid_z[i]))
                push!(k_vals, K_host[i])
            end
        end
        
        CACHE_PTS[surf] = pts
        CACHE_K[surf] = k_vals
    end
    
    CUDA.reclaim()
    println("CUDA geometries cached successfully!")
end

# --- 5. UI DASHBOARD ---
fig = Figure(size = (1600, 1200), backgroundcolor = :gray10)

ax_3d = Axis3(fig[1, 1], title = "3D Area Strain Heatmap", backgroundcolor = :black, aspect = :data)
ax_1d = Axis(fig[1, 2], title = "1D Energy Slice (Fixed A_opt)", 
    xlabel = "Lipid Length (l)", ylabel = "Total Free Energy <E>", 
    backgroundcolor = :black, bottomspinecolor = :white, leftspinecolor = :white,
    xtickcolor = :white, ytickcolor = :white, titlecolor=:cyan, xlabelcolor=:white, ylabelcolor=:white)

# 2D Phase Map Axis
ax_map = Axis(fig[2, 2], title = "2D Thermodynamic Phase Map", 
    xlabel = "Lipid Length (l)", ylabel = "Optimal Area (A_opt)", 
    backgroundcolor = :black, bottomspinecolor = :white, leftspinecolor = :white,
    xtickcolor = :white, ytickcolor = :white, titlecolor=:cyan, xlabelcolor=:white, ylabelcolor=:white)

fig[2, 1] = control_grid = GridLayout(tellwidth = false)

surface_menu = Menu(control_grid[1, 1], options = ["Gyroid (G)", "Primitive (P)", "Diamond (D)"], default = "Gyroid (G)")
Label(control_grid[1, 2], "Domain Bound (π):", color=:white)
bound_slider = Slider(control_grid[1, 3], range = 0.5:0.25:2.0, startvalue = 1.0)

Label(control_grid[2, 1], "Lipid Length (l):", color=:white)
length_slider = Slider(control_grid[2, 2:3], range = 0.1:0.02:1.0, startvalue = 0.5)

Label(control_grid[3, 1], "Optimal Area (A_opt):", color=:white)
area_slider = Slider(control_grid[3, 2:3], range = 0.5:0.02:1.2, startvalue = 0.85)

Label(control_grid[4, 1], "Packing Modulus (k_p):", color=:white)
pack_slider = Slider(control_grid[4, 2:3], range = 0.0:0.05:2.0, startvalue = 0.4)

telemetry_text = Observable("Ready.")
Label(control_grid[5, 1:3], telemetry_text, color = :cyan, fontsize = 18, justification = :left)

export_button = Button(control_grid[6, 1:3], label = "Export 1D Slice to CSV", buttoncolor = :darkred, labelcolor = :white)

# --- 6. OBSERVABLES & RENDERING ---
obs_points = Observable(Point3f[]); obs_energy = Observable(Float32[])
meshscatter!(ax_3d, obs_points, markersize = 0.12, color = obs_energy, 
    colormap = :inferno, colorrange = (0.0, 0.15), shading = true)

l_sweep = collect(0.1:0.02:1.0); num_points = length(l_sweep)
obs_E_G = Observable(zeros(Float32, num_points)); obs_E_D = Observable(zeros(Float32, num_points))
obs_E_P = Observable(zeros(Float32, num_points)); obs_E_L = Observable(zeros(Float32, num_points))
obs_E_HII = Observable(zeros(Float32, num_points))

lines!(ax_1d, l_sweep, obs_E_G, color=:cyan, linewidth=3, label="Gyroid (G)")
lines!(ax_1d, l_sweep, obs_E_D, color=:orange, linewidth=3, label="Diamond (D)")
lines!(ax_1d, l_sweep, obs_E_P, color=:green, linewidth=3, label="Primitive (P)")
lines!(ax_1d, l_sweep, obs_E_L, color=:white, linewidth=3, linestyle=:dash, label="Lamellar (Flat)")
lines!(ax_1d, l_sweep, obs_E_HII, color=:magenta, linewidth=3, linestyle=:dashdot, label="Hexagonal (H_II)")
vlines!(ax_1d, length_slider.value, color=:red, linestyle=:dot, linewidth=2, label="Current State")
axislegend(ax_1d, backgroundcolor=:gray20, labelcolor=:white, position=:lt)

# 2D Map Rendering
a_sweep = collect(0.5:0.02:1.2)
obs_phase_map = Observable(zeros(Float32, length(l_sweep), length(a_sweep)))

# Phase color mapping: 1=Lamellar(Gray), 2=Gyroid(Cyan), 3=Diamond(Orange), 4=Primitive(Green), 5=HII(Magenta)
cmap = cgrad([:gray50, :cyan, :orange, :green, :magenta], 5, categorical=true)
heatmap!(ax_map, l_sweep, a_sweep, obs_phase_map, colormap=cmap, colorrange=(1.0, 5.0))
scatter!(ax_map, length_slider.value, area_slider.value, color=:red, markersize=15, marker=:star5)

# --- 7. UPDATE KERNELS ---
function update_system(surf_type, l_lipid, A_opt, k_p)
    k_vals = CACHE_K[surf_type]
    
    # 3D Heatmap
    obs_points[] = CACHE_PTS[surf_type]
    obs_energy[] = Float32[calc_area_frustration(k, l_lipid, A_opt) for k in k_vals]
    
    # 1D Phase Diagram Update
    obs_E_L[] = Float32[calc_total_energy("Lamellar", [], l, A_opt, k_p) for l in l_sweep]
    obs_E_G[] = Float32[calc_total_energy("Gyroid (G)", CACHE_K["Gyroid (G)"], l, A_opt, k_p) for l in l_sweep]
    obs_E_D[] = Float32[calc_total_energy("Diamond (D)", CACHE_K["Diamond (D)"], l, A_opt, k_p) for l in l_sweep]
    obs_E_P[] = Float32[calc_total_energy("Primitive (P)", CACHE_K["Primitive (P)"], l, A_opt, k_p) for l in l_sweep]
    obs_E_HII[] = Float32[(0.5 * ((1.0 - 0.4f0*l - A_opt)^2)) + (0.5 * k_p * 2.5 * (l^2)) for l in l_sweep]

    e_curr_S = calc_total_energy(surf_type, k_vals, l_lipid, A_opt, k_p)
    telemetry_text[] = "SELECTED: $surf_type | Total Free Energy <E>: $(round(e_curr_S, digits=4))\n"
    
    # Generate 2D Phase Map (Instantaneous on CPU since geometry is cached)
    phase_map = zeros(Float32, length(l_sweep), length(a_sweep))
    for (i, l_val) in enumerate(l_sweep)
        for (j, a_val) in enumerate(a_sweep)
            e_L = calc_total_energy("Lamellar", [], l_val, a_val, k_p)
            e_G = calc_total_energy("Gyroid (G)", CACHE_K["Gyroid (G)"], l_val, a_val, k_p)
            e_D = calc_total_energy("Diamond (D)", CACHE_K["Diamond (D)"], l_val, a_val, k_p)
            e_P = calc_total_energy("Primitive (P)", CACHE_K["Primitive (P)"], l_val, a_val, k_p)
            e_H = (0.5 * ((1.0 - 0.4f0*l_val - a_val)^2)) + (0.5 * k_p * 2.5 * (l_val^2))
            
            # Map index of minimum energy to color code
            phase_map[i, j] = argmin([e_L, e_G, e_D, e_P, e_H])
        end
    end
    obs_phase_map[] = phase_map
end

# --- 8. LISTENERS & INIT ---
on(bound_slider.value) do b_val
    telemetry_text[] = "Re-scanning geometry cache via CUDA..."
    build_geometry_cache!(b_val)
    update_system(surface_menu.selection[], length_slider.value[], area_slider.value[], pack_slider.value[])
end

onany(surface_menu.selection, length_slider.value, area_slider.value, pack_slider.value) do surf, l_val, a_val, kp_val
    update_system(surf, l_val, a_val, kp_val)
end

on(export_button.clicks) do _
    open("thermodynamic_export.csv", "w") do io
        write(io, "Lipid_Length,E_Lamellar,E_Gyroid,E_Diamond,E_Primitive,E_HII\n")
        for i in 1:length(l_sweep)
            write(io, "$(l_sweep[i]),$(obs_E_L[][i]),$(obs_E_G[][i]),$(obs_E_D[][i]),$(obs_E_P[][i]),$(obs_E_HII[][i])\n")
        end
    end
    println("Exported current 1D energy slice to thermodynamic_export.csv")
end

build_geometry_cache!(bound_slider.value[])
update_system(surface_menu.selection[], length_slider.value[], area_slider.value[], pack_slider.value[])

screen = display(fig)
while isopen(screen)
    sleep(0.1)
end
println("Observatory session closed.")
