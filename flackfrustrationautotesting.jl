using GLMakie
using LinearAlgebra
using Statistics
using CUDA
using CUDA.CUFFT
using Optim
using FFTW

# --- 1. NODAL APPROXIMATIONS (GPU COMPATIBLE) ---
const SURF_CODES = Dict("Gyroid (G)" => 1, "Primitive (P)" => 2, "Diamond (D)" => 3)
const PHASE_CODES = Dict("Lamellar"=>1, "Gyroid (G)"=>2, "Diamond (D)"=>3, "Primitive (P)"=>4, "Hexagonal (H_II)"=>5)

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

# --- 2. DIFFERENTIAL GEOMETRY ---
function calc_curvatures_implicit_gpu(type_code::Int, x::Float32, y::Float32, z::Float32, delta::Float32=2f-2)
    Fx = (eval_surface_gpu(type_code, x+delta, y, z) - eval_surface_gpu(type_code, x-delta, y, z)) / (2f0*delta)
    Fy = (eval_surface_gpu(type_code, x, y+delta, z) - eval_surface_gpu(type_code, x, y-delta, z)) / (2f0*delta)
    Fz = (eval_surface_gpu(type_code, x, y, z+delta) - eval_surface_gpu(type_code, x, y, z-delta)) / (2f0*delta)
    
    grad_mag2 = Fx^2 + Fy^2 + Fz^2
    K_out = 0.0f0; H_out = 0.0f0
    
    if grad_mag2 >= 1f-6
        grad_mag = sqrt(grad_mag2)
        Fxx = (eval_surface_gpu(type_code, x+delta, y, z) - 2f0*eval_surface_gpu(type_code, x, y, z) + eval_surface_gpu(type_code, x-delta, y, z)) / (delta^2)
        Fyy = (eval_surface_gpu(type_code, x, y+delta, z) - 2f0*eval_surface_gpu(type_code, x, y, z) + eval_surface_gpu(type_code, x, y-delta, z)) / (delta^2)
        Fzz = (eval_surface_gpu(type_code, x, y, z+delta) - 2f0*eval_surface_gpu(type_code, x, y, z) + eval_surface_gpu(type_code, x, y, z-delta)) / (delta^2)
        
        Fxy = (eval_surface_gpu(type_code, x+delta, y+delta, z) - eval_surface_gpu(type_code, x-delta, y+delta, z) - 
               eval_surface_gpu(type_code, x+delta, y-delta, z) + eval_surface_gpu(type_code, x-delta, y-delta, z)) / (4f0*delta^2)
        Fxz = (eval_surface_gpu(type_code, x+delta, y, z+delta) - eval_surface_gpu(type_code, x-delta, y, z+delta) - 
               eval_surface_gpu(type_code, x+delta, y, z-delta) + eval_surface_gpu(type_code, x-delta, y, z-delta)) / (4f0*delta^2)
        Fyz = (eval_surface_gpu(type_code, x, y+delta, z+delta) - eval_surface_gpu(type_code, x, y-delta, z+delta) - 
               eval_surface_gpu(type_code, x, y+delta, z-delta) + eval_surface_gpu(type_code, x, y-delta, z-delta)) / (4f0*delta^2)

        K_out = (Fx^2*(Fyy*Fzz - Fyz^2) + Fy^2*(Fxx*Fzz - Fxz^2) + Fz^2*(Fxx*Fyy - Fxy^2) +
                2f0*Fx*Fy*(Fxz*Fyz - Fxy*Fzz) + 2f0*Fy*Fz*(Fxy*Fxz - Fyz*Fxx) + 2f0*Fx*Fz*(Fxy*Fyz - Fxz*Fyy)) / (grad_mag2^2)
        H_out = (Fxx*(Fy^2 + Fz^2) + Fyy*(Fx^2 + Fz^2) + Fzz*(Fx^2 + Fy^2) -
                2f0*Fx*Fy*Fxy - 2f0*Fx*Fz*Fxz - 2f0*Fy*Fz*Fyz) / (2f0 * grad_mag2 * grad_mag)
    end
    return K_out, H_out
end

# --- 3. THERMODYNAMICS WITH LIPID SORTING ---
const PACKING_VAR = Dict("Lamellar" => 0.0, "Gyroid (G)" => 1.0, "Diamond (D)" => 1.15, "Primitive (P)"=> 1.65)

function calc_total_energy(surf_type, k_array, h_array, l_lipid, A_opt, k_p, k_a, alpha_sort)
    if isempty(k_array) && surf_type != "Lamellar" return 1e6 end 
    
    var_K = 0.0f0 # Variance of Gaussian curvature
    
    if surf_type == "Lamellar"
        mean_area_E = 0.5 * k_a * ((1.0 - A_opt)^2) 
    else
        total_a_e = 0.0
        for i in 1:length(k_array)
            local_area_factor = (1.0 + 2.0*h_array[i]*l_lipid + (l_lipid^2)*k_array[i])
            total_a_e += 0.5 * k_a * ((local_area_factor - A_opt)^2)
        end
        mean_area_E = total_a_e / length(k_array)
        var_K = var(k_array) # Calculate geometric variance for sorting
    end
    
    packing_E = 0.5 * k_p * PACKING_VAR[surf_type] * (l_lipid^2)
    sorting_E = -1.0 * alpha_sort * var_K # Energy relief from curvature-composition coupling
    
    return mean_area_E + packing_E + sorting_E
end

# --- 4. CUDA CACHE & HIGH-RES SAXS EMULATOR ---
const RES = 128 

const CACHE_PTS = Dict{String, Vector{Point3f}}()
const CACHE_K   = Dict{String, Vector{Float32}}()
const CACHE_H   = Dict{String, Vector{Float32}}()
const CACHE_SAXS= Dict{String, Tuple{Vector{Float32}, Vector{Float32}}}()

function electron_density_gpu(F_val::Float32, c_val::Float32)
    dist = abs(F_val - c_val)
    rho_tail = -1.0f0 * exp(-(dist^2) / 0.02f0)
    rho_head = 1.2f0 * exp(-((dist - 0.3f0)^2) / 0.01f0)
    return rho_head + rho_tail
end

function build_geometry_cache!(b_mult, target_vf)
    empty!(CACHE_PTS); empty!(CACHE_K); empty!(CACHE_H); empty!(CACHE_SAXS)
    bound = Float32(b_mult * π)
    r = Float32.(range(-bound, stop=bound, length=RES))
    
    grid_x = Float32[x for x in r, y in r, z in r][:]
    grid_y = Float32[y for x in r, y in r, z in r][:]
    grid_z = Float32[z for x in r, y in r, z in r][:]
    
    cu_X = CuArray(grid_x); cu_Y = CuArray(grid_y); cu_Z = CuArray(grid_z)
    
    for surf in ["Gyroid (G)", "Primitive (P)", "Diamond (D)"]
        code = SURF_CODES[surf]
        cu_F = eval_surface_gpu.(code, cu_X, cu_Y, cu_Z)
        F_host = Array(cu_F)
        
        c_val = Float32(quantile(F_host, target_vf))
        epsilon = 0.05f0 
        
        K_host = zeros(Float32, length(grid_x))
        H_host = zeros(Float32, length(grid_x))
        pts = Point3f[]; k_vals = Float32[]; h_vals = Float32[]
        
        for i in 1:length(grid_x)
            diff = abs(F_host[i] - c_val)
            if diff < epsilon
                k, h = calc_curvatures_implicit_gpu(code, grid_x[i], grid_y[i], grid_z[i])
                push!(pts, Point3f(grid_x[i], grid_y[i], grid_z[i]))
                push!(k_vals, k); push!(h_vals, h)
            end
        end
        
        CACHE_PTS[surf] = pts; CACHE_K[surf] = k_vals; CACHE_H[surf] = h_vals
        
        cu_density = electron_density_gpu.(cu_F, c_val)
        cu_fft = CUDA.CUFFT.fft(ComplexF32.(cu_density))
        cu_I_q = abs2.(cu_fft) 
        I_q_host = Array(cu_I_q)
        
        freqs = fftfreq(RES) |> Array
        n_bins = 150
        q_max = 0.5f0 
        bins = range(0.01, stop=q_max, length=n_bins)
        
        I_binned = zeros(Float32, n_bins)
        counts = zeros(Int, n_bins)
        
        for k in 1:RES, j in 1:RES, i in 1:RES
            q = sqrt(freqs[i]^2 + freqs[j]^2 + freqs[k]^2)
            if q > 0.01f0 && q < q_max
                bin_idx = searchsortedfirst(bins, q)
                if bin_idx <= n_bins
                    idx = i + (j-1)*RES + (k-1)*RES^2
                    I_binned[bin_idx] += I_q_host[idx]
                    counts[bin_idx] += 1
                end
            end
        end
        
        I_final = I_binned ./ max.(counts, 1)
        CACHE_SAXS[surf] = (collect(bins), I_final)
    end
    CUDA.reclaim()
end

# --- EXPERIMENTAL DATA ---
const EXP_DATA = [(0.35, 0.95, 1), (0.55, 0.85, 2), (0.65, 0.75, 3), (0.85, 0.65, 5)]


# --- 5. UI DASHBOARD ---
fig = Figure(size = (1800, 1300), backgroundcolor = :gray10)

ax_3d = Axis3(fig[1, 1], title = "3D Geometric Strain", backgroundcolor = :black, aspect = :data)
ax_map = Axis(fig[1, 2], title = "2D Phase Map w/ Experimental Data", 
    xlabel = "Lipid Length (l)", ylabel = "Optimal Area (A_opt)", backgroundcolor = :black, 
    bottomspinecolor=:white, leftspinecolor=:white, xtickcolor=:white, ytickcolor=:white, titlecolor=:cyan, xlabelcolor=:white, ylabelcolor=:white)

ax_saxs = Axis(fig[2, 1:2], title = "Theoretical SAXS Diffraction Pattern I(q) [High Res Gaussian Form Factor]", 
    xlabel = "Scattering Vector |q|", ylabel = "Intensity (Log Scale)", yscale=log10, 
    backgroundcolor = :black, bottomspinecolor=:white, leftspinecolor=:white, xtickcolor=:white, ytickcolor=:white, titlecolor=:yellow, xlabelcolor=:white, ylabelcolor=:white)

fig[3, 1:2] = control_grid = GridLayout(tellwidth = false)

surface_menu = Menu(control_grid[1, 1], options = ["Gyroid (G)", "Primitive (P)", "Diamond (D)"], default = "Gyroid (G)")
Label(control_grid[1, 2], "Domain (π):", color=:white)
bound_slider = Slider(control_grid[1, 3], range = 0.5:0.25:2.0, startvalue = 1.0)
Label(control_grid[1, 4], lift(v -> string(round(v, digits=2)), bound_slider.value), color=:cyan, justification=:left)

Label(control_grid[2, 1], "Volume Frac (v_f):", color=:white)
vf_slider = Slider(control_grid[2, 2:3], range = 0.1:0.05:0.9, startvalue = 0.5)
Label(control_grid[2, 4], lift(v -> string(round(v, digits=2)), vf_slider.value), color=:cyan, justification=:left)

Label(control_grid[3, 1], "Lipid Length (l):", color=:white)
length_slider = Slider(control_grid[3, 2:3], range = 0.1:0.02:1.0, startvalue = 0.55)
Label(control_grid[3, 4], lift(v -> string(round(v, digits=2)), length_slider.value), color=:cyan, justification=:left)

Label(control_grid[4, 1], "Optimal Area (A_opt):", color=:white)
area_slider = Slider(control_grid[4, 2:3], range = 0.5:0.02:1.2, startvalue = 0.85)
Label(control_grid[4, 4], lift(v -> string(round(v, digits=2)), area_slider.value), color=:cyan, justification=:left)

Label(control_grid[5, 1], "Packing Modulus (k_p):", color=:white)
pack_slider = Slider(control_grid[5, 2:3], range = 0.0:0.05:2.0, startvalue = 0.4)
Label(control_grid[5, 4], lift(v -> string(round(v, digits=2)), pack_slider.value), color=:cyan, justification=:left)

Label(control_grid[6, 1], "Area Modulus (k_a):", color=:white)
area_mod_slider = Slider(control_grid[6, 2:3], range = 0.1:0.1:2.0, startvalue = 1.0)
Label(control_grid[6, 4], lift(v -> string(round(v, digits=2)), area_mod_slider.value), color=:cyan, justification=:left)

Label(control_grid[7, 1], "Lipid Sorting (α):", color=:yellow)
sort_slider = Slider(control_grid[7, 2:3], range = 0.0:0.02:0.5, startvalue = 0.0)
Label(control_grid[7, 4], lift(v -> string(round(v, digits=2)), sort_slider.value), color=:yellow, justification=:left)

telemetry_text = Observable("Ready.")
# Spanned across all 4 columns to keep it centered and clean
Label(control_grid[8, 1:4], telemetry_text, color = :cyan, fontsize = 18, justification = :left)

fit_button = Button(control_grid[9, 1:4], label = "OPTIM.JL CONTINUOUS INVERSE SOLVER", buttoncolor = :darkgreen, labelcolor = :white)

# --- 6. OBSERVABLES & RENDERING ---
obs_points = Observable(Point3f[]); obs_energy = Observable(Float32[])

meshscatter!(ax_3d, obs_points, markersize = 0.04, color = obs_energy, colormap = :inferno, colorrange = (0.0, 0.15), shading = true)

l_sweep = collect(0.1:0.02:1.0); a_sweep = collect(0.5:0.02:1.2)
obs_phase_map = Observable(zeros(Float32, length(l_sweep), length(a_sweep)))
cmap = cgrad([:gray50, :cyan, :orange, :green, :magenta], 5, categorical=true)
heatmap!(ax_map, l_sweep, a_sweep, obs_phase_map, colormap=cmap, colorrange=(1.0, 5.0))

phase_elements = [PolyElement(color = :gray50), PolyElement(color = :cyan), PolyElement(color = :orange), PolyElement(color = :green), PolyElement(color = :magenta)]
phase_labels = ["Lamellar (Lα)", "Gyroid (Ia3d)", "Diamond (Pn3m)", "Primitive (Im3m)", "Hexagonal (HII)"]
Legend(fig[1, 2], phase_elements, phase_labels, "Thermodynamic Ground State",
    halign = :right, valign = :top, margin = (10, 10, 10, 10),
    tellwidth = false, tellheight = false, backgroundcolor = (:black, 0.7), labelcolor = :white, titlecolor = :cyan, framecolor = :gray30)

exp_l = [d[1] for d in EXP_DATA]; exp_a = [d[2] for d in EXP_DATA]
exp_c = [:white, :cyan, :orange, :magenta]
scatter!(ax_map, exp_l, exp_a, color=exp_c, markersize=20, marker=:circle, strokewidth=2, strokecolor=:black)
scatter!(ax_map, length_slider.value, area_slider.value, color=:red, markersize=15, marker=:star5)

obs_q = Observable(Float32[0.1, 0.2])
obs_Iq = Observable(Float32[1.0, 1.0])
lines!(ax_saxs, obs_q, obs_Iq, color=:yellow, linewidth=2)

# --- 7. UPDATE KERNELS ---
function predict_phase(l_val, a_val, k_p, k_a, alpha_sort)
    e_L = calc_total_energy("Lamellar", [], [], l_val, a_val, k_p, k_a, alpha_sort)
    e_G = calc_total_energy("Gyroid (G)", CACHE_K["Gyroid (G)"], CACHE_H["Gyroid (G)"], l_val, a_val, k_p, k_a, alpha_sort)
    e_D = calc_total_energy("Diamond (D)", CACHE_K["Diamond (D)"], CACHE_H["Diamond (D)"], l_val, a_val, k_p, k_a, alpha_sort)
    e_P = calc_total_energy("Primitive (P)", CACHE_K["Primitive (P)"], CACHE_H["Primitive (P)"], l_val, a_val, k_p, k_a, alpha_sort)
    e_H = (0.5 * k_a * ((1.0 - 0.4f0*l_val - a_val)^2)) + (0.5 * k_p * 2.5 * (l_val^2))
    return argmin([e_L, e_G, e_D, e_P, e_H])
end

function update_system(surf_type, l_lipid, A_opt, k_p, k_a, alpha_sort)
    k_vals = CACHE_K[surf_type]; h_vals = CACHE_H[surf_type]
    
    obs_points[] = CACHE_PTS[surf_type]
    obs_energy[] = Float32[0.5*k_a*((1.0 + 2.0*h_vals[i]*l_lipid + (l_lipid^2)*k_vals[i]) - A_opt)^2 for i in 1:length(k_vals)]
    
    q_bins, I_q = CACHE_SAXS[surf_type]
    obs_q[] = q_bins; obs_Iq[] = I_q .+ 1e-5 
    
    e_curr_S = calc_total_energy(surf_type, k_vals, h_vals, l_lipid, A_opt, k_p, k_a, alpha_sort)
    v_k = isempty(k_vals) ? 0.0 : round(var(k_vals), digits=4)
    telemetry_text[] = "SELECTED: $surf_type | Total Free Energy <E>: $(round(e_curr_S, digits=4)) | Var(K): $v_k\n"
    
    phase_map = zeros(Float32, length(l_sweep), length(a_sweep))
    for (i, l_val) in enumerate(l_sweep)
        for (j, a_val) in enumerate(a_sweep)
            phase_map[i, j] = predict_phase(l_val, a_val, k_p, k_a, alpha_sort)
        end
    end
    obs_phase_map[] = phase_map
end

# --- 8. CONTINUOUS INVERSE SOLVER ---
function continuous_loss(params)
    kp_test, ka_test = params[1], params[2]
    alpha_test = sort_slider.value[] # Respect current sorting strength
    if kp_test < 0.0 || ka_test < 0.1 return 1e9 end 
    
    loss = 0.0
    for (l_val, a_val, true_phase_idx) in EXP_DATA
        e_L = calc_total_energy("Lamellar", [], [], l_val, a_val, kp_test, ka_test, alpha_test)
        e_G = calc_total_energy("Gyroid (G)", CACHE_K["Gyroid (G)"], CACHE_H["Gyroid (G)"], l_val, a_val, kp_test, ka_test, alpha_test)
        e_D = calc_total_energy("Diamond (D)", CACHE_K["Diamond (D)"], CACHE_H["Diamond (D)"], l_val, a_val, kp_test, ka_test, alpha_test)
        e_P = calc_total_energy("Primitive (P)", CACHE_K["Primitive (P)"], CACHE_H["Primitive (P)"], l_val, a_val, kp_test, ka_test, alpha_test)
        e_H = (0.5 * ka_test * ((1.0 - 0.4f0*l_val - a_val)^2)) + (0.5 * kp_test * 2.5 * (l_val^2))
        
        energies = [e_L, e_G, e_D, e_P, e_H]
        e_true = energies[true_phase_idx]
        
        for (i, e) in enumerate(energies)
            if i != true_phase_idx
                loss += max(0.0, e_true - e + 0.01) 
            end
        end
    end
    return loss
end

on(fit_button.clicks) do _
    telemetry_text[] = "Optim.jl: Running Nelder-Mead simplex to solve inverse thermodynamic problem..."
    yield()
    
    initial_x = [pack_slider.value[], area_mod_slider.value[]]
    result = Optim.optimize(continuous_loss, initial_x, NelderMead())
    best_kp, best_ka = Optim.minimizer(result)
    
    set_close_to!(pack_slider, best_kp)
    set_close_to!(area_mod_slider, best_ka)
    telemetry_text[] = "Fit Complete (Nelder-Mead)! Found k_p = $(round(best_kp, digits=3)), k_a = $(round(best_ka, digits=3)). Final Loss = $(round(Optim.minimum(result), digits=4))"
end

# --- 9. LISTENERS & INIT ---
onany(bound_slider.value, vf_slider.value) do b_val, vf_val
    telemetry_text[] = "Re-scanning high-res geometry cache and regenerating FFTs via CUDA..."
    build_geometry_cache!(b_val, vf_val)
    update_system(surface_menu.selection[], length_slider.value[], area_slider.value[], pack_slider.value[], area_mod_slider.value[], sort_slider.value[])
end

onany(surface_menu.selection, length_slider.value, area_slider.value, pack_slider.value, area_mod_slider.value, sort_slider.value) do surf, l_val, a_val, kp_val, ka_val, alpha_val
    update_system(surf, l_val, a_val, kp_val, ka_val, alpha_val)
end

build_geometry_cache!(bound_slider.value[], vf_slider.value[])
update_system(surface_menu.selection[], length_slider.value[], area_slider.value[], pack_slider.value[], area_mod_slider.value[], sort_slider.value[])

screen = display(fig)
while isopen(screen)
    sleep(0.1)
end
