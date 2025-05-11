import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import exp1, k0
from scipy.optimize import brentq
import plotly.graph_objects as go 
import matplotlib.patches as mpatches

# --- Core Hydrogeology Functions ---
def theis_drawdown_unconfined_jacob(Q, K_conductivity, initial_saturated_thickness, Sy_specific_yield, r, t):
    """
    Calculates drawdown using Theis equation with Jacob's correction for unconfined aquifers.
    """
    if K_conductivity <= 0 or initial_saturated_thickness <= 0 or Sy_specific_yield <= 0 or t <= 0:
        return 0.0
    T_initial = K_conductivity * initial_saturated_thickness
    if T_initial <= 0: return 0.0
    if r <= 1e-6: 
        u = ((1e-6)**2 * Sy_specific_yield) / (4 * T_initial * t)
    else: 
        u = (r**2 * Sy_specific_yield) / (4 * T_initial * t)
    
    if u == 0: 
        s0 = np.inf 
    else: 
        s0 = (Q / (4 * np.pi * T_initial)) * exp1(u)

    if s0 >= initial_saturated_thickness * 0.9999: 
        s_corrected = initial_saturated_thickness
    elif s0 <= 0: 
        s_corrected = 0.0
    else:
        s_corrected = s0 - (s0**2) / (2 * initial_saturated_thickness)
        if s_corrected < 0: 
            s_corrected = initial_saturated_thickness
        s_corrected = min(s_corrected, initial_saturated_thickness)
        s_corrected = max(0.0, s_corrected)
    return s_corrected

def calculate_s_p_hantush(Q, T, r, aquifer_thickness, screen_length_actual,
                          d_aq_top_to_screen_top, l_aq_top_to_screen_bottom,
                          z_eff_obs_from_aq_top, num_terms=20):
    """
    Calculates drawdown component due to partial penetration using Hantush (1961, 1964) approximation.
    """
    if T <=0: return 0.0
    if screen_length_actual >= aquifer_thickness - 1e-3: return 0.0 
    if screen_length_actual <= 0 or aquifer_thickness <= 0: return 0.0
    
    s_p_sum = 0.0
    b = aquifer_thickness 
    Ls = screen_length_actual

    d_param = d_aq_top_to_screen_top 
    l_param = l_aq_top_to_screen_bottom
    
    for n in range(1, num_terms + 1):
        term_sin = (np.sin(n * np.pi * l_param / b) - np.sin(n * np.pi * d_param / b))
        term_cos = np.cos(n * np.pi * z_eff_obs_from_aq_top / b)
        
        k0_arg = (n * np.pi * r) / b
        if k0_arg < 1e-9: 
            term_k0 = -np.log(k0_arg/2.0) - 0.5772156649 if r > 1e-10 else np.log(2/(n*np.pi*(1e-10/b))) - 0.5772156649
        elif k0_arg > 35: 
            term_k0 = 0.0
        else:
            term_k0 = k0(k0_arg)
            
        s_p_sum += (1/n) * term_sin * term_cos * term_k0
        
    factor = (Q / (2 * np.pi * T)) * ( (2 * b) / (np.pi * Ls) ) if Ls > 0 else 0
    return factor * s_p_sum

def calculate_s_p_hantush_for_unconfined(Q, K_conductivity, initial_saturated_thickness, r,
                                         screen_length_actual, d_initial_wt_to_screen_top,
                                         l_initial_wt_to_screen_bottom, z_mid_screen_from_initial_wt, num_terms=20):
    """
    Wrapper for Hantush partial penetration calculation adapted for unconfined conditions.
    """
    T_initial = K_conductivity * initial_saturated_thickness
    if T_initial <=0: return 0.0
    return calculate_s_p_hantush(Q, T_initial, r, initial_saturated_thickness, screen_length_actual,
                                d_initial_wt_to_screen_top, l_initial_wt_to_screen_bottom,
                                z_mid_screen_from_initial_wt, num_terms)

def advanced_drawdown_solution_unconfined(Q, K_conductivity, initial_saturated_thickness, Sy_specific_yield, r, t,
                                          screen_length_actual, d_initial_wt_to_screen_top,
                                          l_initial_wt_to_screen_bottom, z_mid_screen_from_initial_wt,
                                          apply_partial_penetration=True):
    """
    Calculates total drawdown by summing Theis-Jacob and Hantush partial penetration components.
    """
    s_jacob = theis_drawdown_unconfined_jacob(Q, K_conductivity, initial_saturated_thickness, Sy_specific_yield, r, t)
    
    s_p_approx = 0.0
    if apply_partial_penetration and screen_length_actual < initial_saturated_thickness - 1e-3 and screen_length_actual > 1e-3 :
        s_p_approx = calculate_s_p_hantush_for_unconfined(Q, K_conductivity, initial_saturated_thickness, r,
                                                          screen_length_actual, d_initial_wt_to_screen_top,
                                                          l_initial_wt_to_screen_bottom, z_mid_screen_from_initial_wt)
    
    total_drawdown = s_jacob + s_p_approx
    return min(total_drawdown, initial_saturated_thickness)

def calculate_superposition_unconfined(x_obs, y_obs, well_details_list, K_conductivity,
                                       initial_saturated_thickness, Sy_specific_yield, t,
                                       initial_water_table_abs_elev, borehole_dia, apply_partial_penetration=True):
    """
    Calculates total drawdown at an observation point due to multiple wells using superposition.
    """
    total_s = 0.0
    for well_info in well_details_list:
        well_x, well_y = well_info['coords']
        Q_well = well_info['Q']
        screen_top_elev_well = well_info['screen_top_elev'] 
        screen_length_well = well_info['screen_length']

        r_obs_to_well = np.sqrt((x_obs - well_x)**2 + (y_obs - well_y)**2)
        
        current_r_for_calc = r_obs_to_well
        if r_obs_to_well < (borehole_dia / 2.0) * 0.999: 
            current_r_for_calc = borehole_dia / 2.0 

        d_initial_wt_to_screen_top_well = initial_water_table_abs_elev - screen_top_elev_well
        screen_bottom_elev_well = screen_top_elev_well - screen_length_well 
        l_initial_wt_to_screen_bottom_well = initial_water_table_abs_elev - screen_bottom_elev_well

        d_eff = max(0.0, d_initial_wt_to_screen_top_well)
        d_eff = min(d_eff, initial_saturated_thickness) 
        
        l_eff = max(d_eff, l_initial_wt_to_screen_bottom_well) 
        l_eff = min(l_eff, initial_saturated_thickness) 

        effective_screen_length_for_sp = l_eff - d_eff 
        
        apply_specific_well_partial_penetration = apply_partial_penetration and (effective_screen_length_for_sp > 1e-3) and (effective_screen_length_for_sp < initial_saturated_thickness - 1e-3)

        mid_screen_depth_from_initial_wt = d_eff + (effective_screen_length_for_sp / 2.0)
        mid_screen_depth_from_initial_wt = max(0.0, min(mid_screen_depth_from_initial_wt, initial_saturated_thickness))

        s_individual = advanced_drawdown_solution_unconfined(
            Q_well, K_conductivity, initial_saturated_thickness, Sy_specific_yield, current_r_for_calc, t,
            effective_screen_length_for_sp, d_eff, l_eff, mid_screen_depth_from_initial_wt,
            apply_partial_penetration=apply_specific_well_partial_penetration)
        total_s += s_individual
        
    return min(total_s, initial_saturated_thickness)

def objective_func_target_time_unconfined(t, x_obs, y_obs, s_target_val, well_details_list, K_conductivity,
                                           initial_saturated_thickness, Sy_specific_yield, initial_water_table_abs_elev, borehole_dia):
    """
    Objective function for root finding: current drawdown - target drawdown.
    """
    if t <= 0: return s_target_val * 1e6 
    s_current = calculate_superposition_unconfined(x_obs, y_obs, well_details_list, K_conductivity,
                                                  initial_saturated_thickness, Sy_specific_yield, t,
                                                  initial_water_table_abs_elev, borehole_dia, apply_partial_penetration=True)
    return s_current - s_target_val

def find_time_to_target_drawdown_unconfined(x_obs, y_obs, s_target_val, well_details_list, K_conductivity,
                                            initial_saturated_thickness, Sy_specific_yield, initial_water_table_abs_elev, borehole_dia,
                                            min_time_search=1e-5, max_time_search=10000, bracket_factor=2, max_bracket_iter=30):
    """
    Finds the time required to reach a target drawdown value at a specific observation point.
    """
    if s_target_val <= 0: 
        return "Target drawdown must be positive."
    
    original_s_target = s_target_val
    if s_target_val >= initial_saturated_thickness * 0.999: 
        st.warning(f"Original target drawdown ({original_s_target:.2f}m) is very close to or exceeds initial saturated thickness ({initial_saturated_thickness:.2f}m). Adjusting target to {initial_saturated_thickness * 0.999:.2f}m for calculation.")
        s_target_val = initial_saturated_thickness * 0.999
    
    if s_target_val <=0: 
        return "Adjusted target drawdown is not positive."

    args = (x_obs, y_obs, s_target_val, well_details_list, K_conductivity, initial_saturated_thickness, Sy_specific_yield, initial_water_table_abs_elev, borehole_dia)
    
    f_at_min_time = objective_func_target_time_unconfined(min_time_search, *args)
    if f_at_min_time >= 0: 
         s_actual_at_min_time = f_at_min_time + s_target_val
         return f"Target drawdown ({original_s_target:.2f}m, adj. to {s_target_val:.2f}m) likely reached at or before {min_time_search:.2e} days (drawdown at min_time = {s_actual_at_min_time:.2f}m)."

    t_low, f_low = min_time_search, f_at_min_time
    t_high = t_low
    for i in range(max_bracket_iter):
        t_high *= bracket_factor
        if t_high > max_time_search:
            s_max_t = objective_func_target_time_unconfined(max_time_search, *args) + s_target_val
            return f"Target drawdown ({original_s_target:.2f}m, adj. to {s_target_val:.2f}m) not reached by max search time ({max_time_search} days). Max drawdown achieved = {s_max_t:.2f}m."
        
        f_high = objective_func_target_time_unconfined(t_high, *args)
        if f_high >= 0: 
            try:
                return brentq(objective_func_target_time_unconfined, t_low, t_high, args=args, xtol=1e-6, rtol=1e-6)
            except ValueError as e:
                return f"Root finding error with brentq: {e}. Bracket: [{t_low:.3e}, {t_high:.3e}], Values: [{f_low:.3e}, {f_high:.3e}]"
        
        t_low, f_low = t_high, f_high 

    s_max_brack_t = objective_func_target_time_unconfined(t_high, *args) + s_target_val
    return f"Target drawdown ({original_s_target:.2f}m, adj. to {s_target_val:.2f}m) not reached (bracketing failed to find upper bound up to {t_high:.2e} days). Drawdown at this time = {s_max_brack_t:.2f}m."


# --- Main Streamlit App Function ---
def main():
    st.set_page_config(page_title="Dewatering Simulator", layout="wide")
    st.title("Interactive Dewatering System Simulator (Unconfined Aquifer Approx.)")
    st.markdown("""
    This application simulates dewatering from an unconfined aquifer using multiple wells.
    It uses the Theis equation with Jacob's correction for unconfined conditions and
    an approximated Hantush correction for partial penetration effects.
    **Note:** This model has limitations, especially for very large drawdowns or complex geology.
    The partial penetration term is an approximation based on confined aquifer theory.
    """)

    st.sidebar.header("Input Parameters")

    # Define default values once
    Q_rate_default_val_Ls = 200.0 / 86.4 
    borehole_diameter_default_val = 0.3 
    well_depth_from_ground_default_val = 21.0 
    screen_length_design_default_val = 18.0 
    excavation_center_x_default_val = 50.0 
    excavation_center_y_default_val = 0.0 
    excavation_length_default_val = 40.0 
    excavation_width_default_val = 20.0 
    well_offset_from_excavation_default_val = 5.0 
    num_wells_along_length_default_val = 3 
    num_wells_along_width_default_val = 0 
    target_water_level_default_val = -15.0 
    time_to_target_default_value = 10.0 

    # --- Aquifer Properties ---
    st.sidebar.markdown("---") # Visual separator
    st.sidebar.markdown("**Aquifer Properties:**")
    K_conductivity_ui_ms = st.sidebar.number_input("Hydraulic Conductivity (K, m/s)", value=10.0/86400, min_value=0.01/86400, max_value=1000.0/86400, step=0.1/86400, format="%.2e", key="k_ms_sidebar")
    K_conductivity_ui = K_conductivity_ui_ms * 86400 
    Sy_specific_yield_ui = st.sidebar.number_input("Specific Yield (Sy)", value=0.15, min_value=0.01, max_value=0.5, step=0.01, format="%.2f", key="sy_sidebar")
    ground_level_elev_ui = st.sidebar.number_input("Ground Level Elevation (m)", value=0.0, step=0.5, format="%.1f", key="gl_sidebar")
    initial_water_table_elev_ui = st.sidebar.number_input("Initial Water Table Elevation (m)", value=-2.0, step=0.5, format="%.1f", key="iwt_sidebar")
    aquifer_base_elev_ui = st.sidebar.number_input("Aquifer Base Elevation (m)", value=-22.0, step=0.5, format="%.1f", key="ab_sidebar")

    if initial_water_table_elev_ui <= aquifer_base_elev_ui:
        st.sidebar.error("Error: Initial water table must be above aquifer base elevation.")
        st.stop()
    b_initial_saturated_thickness_ui = initial_water_table_elev_ui - aquifer_base_elev_ui
    st.sidebar.markdown(f"*Derived Initial Saturated Thickness: **{b_initial_saturated_thickness_ui:.2f} m***")

    # --- Default Well Design ---
    st.sidebar.markdown("---") # Visual separator
    st.sidebar.markdown("**Default Well Design:**")
    Q_rate_input_Ls_ui = st.sidebar.number_input(
        "Pumping Rate per Well (Q, L/s)", 
        value=Q_rate_default_val_Ls, 
        min_value=0.01, 
        step=0.1, 
        format="%.2f", 
        key="q_rate_sidebar_ls_v3" 
    )
    Q_rate_default_ui = Q_rate_input_Ls_ui * 86.4 
    Q_rate_gpm_display = Q_rate_input_Ls_ui * 15.850323 
    st.sidebar.markdown(
        f"<small>Equivalent to: {Q_rate_default_ui:.2f} m³/day &nbsp;&nbsp;|&nbsp;&nbsp; {Q_rate_gpm_display:.2f} GPM (US)</small>", 
        unsafe_allow_html=True
    )
    borehole_diameter_ui = st.sidebar.number_input("Borehole Diameter (m)", value=borehole_diameter_default_val, min_value=0.05, max_value=2.0, step=0.01, format="%.2f", key="borehole_dia_sidebar_v3")
    well_depth_from_ground_ui = st.sidebar.number_input("Well Depth from Ground (m)", value=well_depth_from_ground_default_val, min_value=1.0, step=0.5, format="%.1f", key="well_depth_sidebar_v3")
    screen_length_design_ui = st.sidebar.number_input("Screen Length Design (m)", value=screen_length_design_default_val, min_value=1.0, step=0.5, format="%.1f", key="screen_len_sidebar_v3")

    # --- Well Layout (Perimeter around Excavation) ---
    st.sidebar.markdown("---") # Visual separator
    st.sidebar.markdown("**Well Layout (Perimeter around Excavation):**")
    excavation_center_x_ui = st.sidebar.number_input("Excavation Center X (m)", value=excavation_center_x_default_val, step=5.0, format="%.1f", key="exc_center_x_sidebar_v3")
    excavation_center_y_ui = st.sidebar.number_input("Excavation Center Y (m)", value=excavation_center_y_default_val, step=5.0, format="%.1f", key="exc_center_y_sidebar_v3")
    excavation_length_ui = st.sidebar.number_input("Excavation Length (along X, m)", value=excavation_length_default_val, min_value=1.0, step=1.0, format="%.1f", key="exc_len_sidebar_v3")
    excavation_width_ui = st.sidebar.number_input("Excavation Width (along Y, m)", value=excavation_width_default_val, min_value=1.0, step=1.0, format="%.1f", key="exc_width_sidebar_v3")
    well_offset_from_excavation_ui = st.sidebar.number_input("Well Offset from Excavation Edge (m)", value=well_offset_from_excavation_default_val, min_value=0.0, step=0.5, format="%.1f", key="well_offset_sidebar_v3")
    num_wells_along_length_ui = st.sidebar.number_input("Num Wells on One Long Side (N & S)", min_value=1, max_value=50, value=num_wells_along_length_default_val, step=1, key="num_wells_l_sidebar_ni_v3")
    num_wells_along_width_ui = st.sidebar.number_input("Num Wells on One Short Side (E & W, if >0)", min_value=0, max_value=50, value=num_wells_along_width_default_val, step=1, key="num_wells_w_sidebar_ni_v3")

    # --- Helper function to generate and process well data ---
    def generate_well_details(num_l, num_w, exc_cx, exc_cy, exc_l, exc_w, offset, 
                              q_rate, well_depth_fg, screen_len_design, 
                              ground_elev, init_wt_elev, aquifer_base_elev, prefix="Well"):
        wells_temp_list = []
        spacing_L = exc_l / (num_l - 1) if num_l > 1 else 0 
        for i in range(num_l):
            if num_l == 1: 
                x_coord = exc_cx
            else: 
                x_coord = (exc_cx - exc_l / 2) + (i * spacing_L)
            
            wells_temp_list.append({'id': f'{prefix}_N{i+1}', 
                                    'coords': (x_coord, exc_cy + exc_w / 2 + offset), 
                                    'Q': q_rate, 'well_depth_from_ground': well_depth_fg, 
                                    'screen_length_design': screen_len_design})
            wells_temp_list.append({'id': f'{prefix}_S{i+1}', 
                                    'coords': (x_coord, exc_cy - exc_w / 2 - offset), 
                                    'Q': q_rate, 'well_depth_from_ground': well_depth_fg, 
                                    'screen_length_design': screen_len_design})

        if num_w > 0:
            spacing_W = exc_w / (num_w - 1) if num_w > 1 else 0
            for i in range(num_w):
                if num_w == 1: 
                    y_coord = exc_cy
                else: 
                    y_coord = (exc_cy - exc_w / 2) + (i * spacing_W)
                
                is_corner_well = (num_l > 0) and \
                                 ( (i == 0 and abs(y_coord - (exc_cy - exc_w/2)) < 1e-3) or \
                                   (i == num_w -1 and abs(y_coord - (exc_cy + exc_w/2)) < 1e-3) )


                if not is_corner_well:
                    wells_temp_list.append({'id': f'{prefix}_E{i+1}', 
                                            'coords': (exc_cx + exc_l / 2 + offset, y_coord), 
                                            'Q': q_rate, 'well_depth_from_ground': well_depth_fg, 
                                            'screen_length_design': screen_len_design})
                    wells_temp_list.append({'id': f'{prefix}_W{i+1}', 
                                            'coords': (exc_cx - exc_l / 2 - offset, y_coord), 
                                            'Q': q_rate, 'well_depth_from_ground': well_depth_fg, 
                                            'screen_length_design': screen_len_design})
        
        processed_wells = []
        seen_coords = set()
        for well in wells_temp_list:
            coord_tuple = tuple(well['coords'])
            if coord_tuple not in seen_coords:
                well_bottom_abs_elev = ground_elev - well['well_depth_from_ground']
                screen_top_elev_design = well_bottom_abs_elev + well['screen_length_design']
                
                actual_screen_top_elev = min(screen_top_elev_design, init_wt_elev) 
                actual_screen_bottom_elev = max(well_bottom_abs_elev, aquifer_base_elev) 

                well['screen_top_elev'] = actual_screen_top_elev
                well['screen_bottom_elev'] = actual_screen_bottom_elev
                
                well['screen_length'] = actual_screen_top_elev - actual_screen_bottom_elev
                if well['screen_length'] < 1e-3: 
                    well['screen_length'] = 1e-3 
                    if actual_screen_top_elev < actual_screen_bottom_elev + 1e-3:
                         well['screen_top_elev'] = actual_screen_bottom_elev + 1e-3

                if well['screen_length'] > 1e-3 : 
                    processed_wells.append(well)
                    seen_coords.add(coord_tuple)
        return processed_wells

    wells_data_for_time_calc = generate_well_details(
        num_wells_along_length_ui, num_wells_along_width_ui,
        excavation_center_x_ui, excavation_center_y_ui,
        excavation_length_ui, excavation_width_ui,
        well_offset_from_excavation_ui, Q_rate_default_ui,
        well_depth_from_ground_ui, screen_length_design_ui,
        ground_level_elev_ui, initial_water_table_elev_ui, aquifer_base_elev_ui,
        prefix="WellTime"
    )
    
    obs_point_for_time_calc_coords = (excavation_center_x_ui, excavation_center_y_ui)

    # --- Dewatering Design & Plot Time ---
    st.sidebar.markdown("---") # Visual separator
    st.sidebar.markdown("**Dewatering Design:**")
    target_water_level_final_ui = st.sidebar.number_input("Target Water Level (m elev.)", value=target_water_level_default_val, step=0.5, format="%.1f", key="target_wl_sidebar_v3")
    if target_water_level_final_ui >= initial_water_table_elev_ui:
        st.sidebar.warning("Target water level is at or above initial water table. No dewatering needed or target is invalid.")

    calculated_time_to_target = time_to_target_default_value 
    target_drawdown_for_calc = initial_water_table_elev_ui - target_water_level_final_ui
    
    condition_for_time_calc = (target_drawdown_for_calc > 0) or \
                              (target_drawdown_for_calc == 0 and target_water_level_final_ui == initial_water_table_elev_ui)

    if wells_data_for_time_calc and condition_for_time_calc:
        if not (target_drawdown_for_calc == 0 and target_water_level_final_ui == initial_water_table_elev_ui):
            with st.spinner("Estimating time to target..."):
                time_result = find_time_to_target_drawdown_unconfined(
                    obs_point_for_time_calc_coords[0], obs_point_for_time_calc_coords[1],
                    target_drawdown_for_calc, wells_data_for_time_calc, 
                    K_conductivity_ui, 
                    b_initial_saturated_thickness_ui, 
                    Sy_specific_yield_ui, 
                    initial_water_table_elev_ui, 
                    borehole_diameter_ui
                )
            if isinstance(time_result, (int, float)):
                calculated_time_to_target = float(time_result)
            else:
                st.sidebar.caption(str(time_result)) 
    
    calculated_time_to_target = max(0.01, min(calculated_time_to_target, 1000.0)) 

    t_pumping_snapshot_ui = st.sidebar.number_input("Time for Plot Snapshot (days)", min_value=0.01, max_value=1000.0, value=calculated_time_to_target, step=0.1, format="%.2f", key="time_input_sidebar_v3", help="Time at which drawdown is calculated for plots and analysis. Default is estimated time to target.")
    
    st.sidebar.markdown("---") 
    st.sidebar.markdown("**Plotting Options:**")
    y_plot_slice_ui = st.sidebar.number_input("Y-coordinate for 1D Profile Slice (m)", value=excavation_center_y_ui, step=1.0, format="%.1f", key="y_slice_1d_sidebar_v3")
    resolution_3d_ui = st.sidebar.slider("3D Grid Resolution (nx=ny)", min_value=20, max_value=70, value=30, step=5, key="res_3d_sidebar_v3")
    st.sidebar.markdown("---") # Visual separator at the end of sidebar inputs


    wells_data_ui = generate_well_details(
        num_wells_along_length_ui, num_wells_along_width_ui,
        excavation_center_x_ui, excavation_center_y_ui,
        excavation_length_ui, excavation_width_ui,
        well_offset_from_excavation_ui, Q_rate_default_ui,
        well_depth_from_ground_ui, screen_length_design_ui,
        ground_level_elev_ui, initial_water_table_elev_ui, aquifer_base_elev_ui,
        prefix="Well"
    )
        
    if not wells_data_ui: 
        st.error("No wells defined or all defined wells have zero effective screen length. Adjust layout or well design parameters.")
        st.stop()

    for well in wells_data_ui:
        if well['screen_length'] <= 1e-2 and well['screen_length'] > 1e-3: 
             st.warning(f"Screen length for {well['id']} is very small ({well['screen_length']:.3f}m). Check well design and aquifer geometry.", icon="⚠️")

    observation_points_to_check_ui = [
        {'name': 'Excavation Center', 'coords': (excavation_center_x_ui, excavation_center_y_ui)},
        {'name': 'Mid E Edge of Excav.', 'coords': (excavation_center_x_ui + excavation_length_ui/2, excavation_center_y_ui)},
        {'name': 'Mid N Edge of Excav.', 'coords': (excavation_center_x_ui, excavation_center_y_ui + excavation_width_ui/2)},
        {'name': 'NE Corner of Excav.', 'coords': (excavation_center_x_ui + excavation_length_ui/2, excavation_center_y_ui + excavation_width_ui/2)},
    ]

    # --- Main Panel ---
    st.header("Simulation Outputs")

    # --- 1D Plot ---
    st.subheader("1D Dewatering Profile")
    excav_x_min = excavation_center_x_ui - excavation_length_ui / 2
    excav_x_max = excavation_center_x_ui + excavation_length_ui / 2
    well_x_coords = [wp['coords'][0] for wp in wells_data_ui] if wells_data_ui else [excav_x_min, excav_x_max]
    
    plot_margin_1d = max(50, excavation_length_ui * 0.5) 
    x_plot_min_1d = min(well_x_coords) - plot_margin_1d 
    x_plot_max_1d = max(well_x_coords) + plot_margin_1d
    
    x_plot_1d = np.linspace(x_plot_min_1d, x_plot_max_1d, 200)
    
    drawdown_values_unconfined_1d = np.array([calculate_superposition_unconfined(xi, y_plot_slice_ui, wells_data_ui, K_conductivity_ui, b_initial_saturated_thickness_ui, Sy_specific_yield_ui, t_pumping_snapshot_ui, initial_water_table_elev_ui, borehole_diameter_ui, True) for xi in x_plot_1d])
    water_table_final_1d = initial_water_table_elev_ui - drawdown_values_unconfined_1d
    
    fig1d, ax1d = plt.subplots(1, 1, figsize=(10, 7)) 
    ax1d.plot(x_plot_1d, water_table_final_1d, 'b-', label='Water Table (Unconf. Approx. + sP)')
    
    drawdown_values_unconf_no_sp_1d = np.array([calculate_superposition_unconfined(xi, y_plot_slice_ui, wells_data_ui, K_conductivity_ui, b_initial_saturated_thickness_ui, Sy_specific_yield_ui,t_pumping_snapshot_ui, initial_water_table_elev_ui, borehole_diameter_ui, False) for xi in x_plot_1d])
    ax1d.plot(x_plot_1d, initial_water_table_elev_ui - drawdown_values_unconf_no_sp_1d, 'c--', label='Water Table (Unconf. Approx. no sP)', alpha=0.7)
    
    ax1d.axhline(y=ground_level_elev_ui, color='k', linestyle='-', label='Ground Level')
    ax1d.axhline(y=initial_water_table_elev_ui, color='g', linestyle='--', label='Initial Water Table')
    ax1d.axhline(y=target_water_level_final_ui, color='r', linestyle='-.', linewidth=1.2, label='Target Water Level') 
    
    ax1d.fill_between(x_plot_1d, aquifer_base_elev_ui, initial_water_table_elev_ui, color='sandybrown', alpha=0.3, label='Unconf. Aquifer (Initial Sat.)')
    ax1d.axhline(y=aquifer_base_elev_ui, color='saddlebrown', linestyle='-', linewidth=1.5, label=f'Aquifer Base')
    
    well_labels_added = {'casing': False, 'screen': False} # Removed 'location'
    y_slice_tolerance = max(excavation_width_ui * 0.25, 5.0) + well_offset_from_excavation_ui 

    for well_info in wells_data_ui:
        well_x_coord = well_info['coords'][0]
        if abs(well_info['coords'][1] - y_plot_slice_ui) < y_slice_tolerance or len(wells_data_ui) <=4 :
            well_screen_top_elev_plot = well_info['screen_top_elev']
            well_screen_bottom_elev_plot = well_info['screen_bottom_elev']
            
            casing_top_for_plot = ground_level_elev_ui 
            casing_bottom_for_plot = well_screen_top_elev_plot 
            
            ax1d.plot([well_x_coord, well_x_coord], [casing_top_for_plot, casing_bottom_for_plot], 
                      color='dimgray', linestyle='-', linewidth=2, alpha=0.8, 
                      label='Well Casing' if not well_labels_added['casing'] else "")
            if not well_labels_added['casing']: well_labels_added['casing'] = True
            
            ax1d.plot([well_x_coord, well_x_coord], [well_screen_top_elev_plot, well_screen_bottom_elev_plot], 
                      color='darkblue', linestyle='--', dashes=(3,3), linewidth=2.5, 
                      label='Well Screen' if not well_labels_added['screen'] else "")
            if not well_labels_added['screen']: well_labels_added['screen'] = True
            
            # Removed the scatter plot for well location marker (triangle)
            # ax1d.scatter([well_x_coord], [ground_level_elev_ui], color='darkred', s=50, zorder=10, 
            #              marker='v', edgecolors='black',
            #              label='Well Location' if not well_labels_added['location'] else "")
            # if not well_labels_added['location']: well_labels_added['location'] = True

    ax1d.set_ylabel('Elevation (m)')
    ax1d.set_xlabel(f'Distance (m) along X-axis (Profile at Y = {y_plot_slice_ui:.1f}m)')
    title_str_1d = (f'1D Dewatering Profile at t={t_pumping_snapshot_ui:.2f} days\n'
                    f'K={K_conductivity_ui:.2e} m/d, Init. Sat. Thick.={b_initial_saturated_thickness_ui:.1f}m, Sy={Sy_specific_yield_ui:.2f}')
    ax1d.set_title(title_str_1d, fontsize=10)
    ax1d.grid(True, linestyle=':', alpha=0.7)
    
    handles, labels = ax1d.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) 
    # Adjusted bbox_to_anchor to move legend further down, and increased bottom margin with subplots_adjust
    ax1d.legend(by_label.values(), by_label.keys(), fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.30), ncol=3) 
    
    finite_wt_final_1d = water_table_final_1d[np.isfinite(water_table_final_1d)]
    min_y_data_1d = min(finite_wt_final_1d) if len(finite_wt_final_1d) > 0 else initial_water_table_elev_ui
    min_y_ax1d = min(min_y_data_1d - 2, aquifer_base_elev_ui - 2, target_water_level_final_ui -2) 
    max_y_ax1d = ground_level_elev_ui + 2 
    ax1d.set_ylim(min_y_ax1d, max_y_ax1d)
    
    fig1d.subplots_adjust(bottom=0.32) # Increased bottom margin
    st.pyplot(fig1d)

    # --- 3D Plot ---
    st.subheader("3D Dewatering Visualization")
    nx_3d, ny_3d = resolution_3d_ui, resolution_3d_ui

    with st.spinner(f"Calculating water table for {nx_3d*ny_3d} points for 3D plot..."):
        x_coords_wells_3d = [w['coords'][0] for w in wells_data_ui] if wells_data_ui else [excavation_center_x_ui]
        y_coords_wells_3d = [w['coords'][1] for w in wells_data_ui] if wells_data_ui else [excavation_center_y_ui]
        
        margin_3d = max(30, excavation_length_ui*0.3, excavation_width_ui*0.3) 
        x_min_3d = min(x_coords_wells_3d) - margin_3d 
        x_max_3d = max(x_coords_wells_3d) + margin_3d
        y_min_3d = min(y_coords_wells_3d) - margin_3d
        y_max_3d = max(y_coords_wells_3d) + margin_3d
        
        X_3d_vals = np.linspace(x_min_3d, x_max_3d, nx_3d)
        Y_3d_vals = np.linspace(y_min_3d, y_max_3d, ny_3d)
        X_3d_grid, Y_3d_grid = np.meshgrid(X_3d_vals, Y_3d_vals)
        
        Z_water_table_3d = np.zeros_like(X_3d_grid)
        for i in range(X_3d_grid.shape[0]):
            for j in range(X_3d_grid.shape[1]):
                x_obs, y_obs = X_3d_grid[i, j], Y_3d_grid[i, j]
                drawdown = calculate_superposition_unconfined(x_obs, y_obs, wells_data_ui, K_conductivity_ui, b_initial_saturated_thickness_ui, Sy_specific_yield_ui, t_pumping_snapshot_ui, initial_water_table_elev_ui, borehole_diameter_ui, True)
                Z_water_table_3d[i, j] = initial_water_table_elev_ui - drawdown
                if Z_water_table_3d[i, j] < aquifer_base_elev_ui: 
                    Z_water_table_3d[i, j] = aquifer_base_elev_ui 

        color_metric_3d = np.where(Z_water_table_3d > target_water_level_final_ui, 1, 0) 

        fig3d_plotly = go.Figure()
        fig3d_plotly.add_trace(go.Surface(
            x=X_3d_grid, y=Y_3d_grid, z=Z_water_table_3d,
            surfacecolor=color_metric_3d,
            colorscale=[[0, 'cornflowerblue'], [1, 'salmon']], 
            cmin=0, cmax=1,
            showscale=False, 
            opacity=0.85,
            name='Final Water Table'
        ))
        fig3d_plotly.add_trace(go.Surface(
            x=X_3d_grid, y=Y_3d_grid, z=np.full_like(X_3d_grid, initial_water_table_elev_ui),
            colorscale=[[0,'rgba(0,200,0,0.2)'],[1,'rgba(0,200,0,0.2)']], 
            showscale=False, opacity=0.3, name='Initial Water Table',
            hoverinfo='skip'
        ))
        fig3d_plotly.add_trace(go.Surface(
            x=X_3d_grid, y=Y_3d_grid, z=np.full_like(X_3d_grid, aquifer_base_elev_ui),
            colorscale=[[0,'rgba(160,82,45,0.3)'],[1,'rgba(160,82,45,0.3)']], 
            showscale=False, opacity=0.4, name='Aquifer Base',
            hoverinfo='skip'
        ))

        for well_idx, well in enumerate(wells_data_ui): 
            wx, wy = well['coords']
            screen_top_elev = well['screen_top_elev']
            screen_bottom_elev = well['screen_bottom_elev']
            casing_top_elev = ground_level_elev_ui
            
            # Set showlegend to False for well parts to remove them from the legend
            fig3d_plotly.add_trace(go.Scatter3d(x=[wx, wx], y=[wy, wy], z=[casing_top_elev, screen_top_elev], 
                                                mode='lines', line=dict(color='black', width=6), 
                                                name='Well Casing', legendgroup='well_parts', 
                                                showlegend=False)) # Changed to False
            fig3d_plotly.add_trace(go.Scatter3d(x=[wx, wx], y=[wy, wy], z=[screen_top_elev, screen_bottom_elev], 
                                                mode='lines', line=dict(color='blue', width=6, dash='dash'), 
                                                name='Well Screen', legendgroup='well_parts', 
                                                showlegend=False)) # Changed to False
        
        min_z_3d = min(aquifer_base_elev_ui - 1, np.min(Z_water_table_3d) - 1 if Z_water_table_3d.size > 0 else aquifer_base_elev_ui -1)
        max_z_3d = ground_level_elev_ui + 1
        
        fig3d_plotly.update_layout(
            title=f'3D Dewatered Surface (t={t_pumping_snapshot_ui:.2f} days) vs Target ({target_water_level_final_ui:.1f}m elev.)',
            scene=dict(
                xaxis_title='X Distance (m)',
                yaxis_title='Y Distance (m)',
                zaxis_title='Elevation (m)',
                zaxis=dict(range=[min_z_3d, max_z_3d]),
                aspectmode='cube', 
                camera=dict(eye=dict(x=1.5, y=-1.75, z=0.8)) 
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.7)'),
            height=700
        )
        # These legend items for surface colors remain
        fig3d_plotly.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', 
                                           marker=dict(color='salmon', size=10, symbol='square'), 
                                           name=f'WT > Target ({target_water_level_final_ui:.1f}m)'))
        fig3d_plotly.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', 
                                           marker=dict(color='cornflowerblue', size=10, symbol='square'), 
                                           name=f'WT ≤ Target ({target_water_level_final_ui:.1f}m)'))

        st.plotly_chart(fig3d_plotly, use_container_width=True)

    # --- Text Outputs (Dewatering System Analysis & Time to Target) ---
    st.subheader("Dewatering System Analysis")
    st.markdown(f"**Analysis at t = {t_pumping_snapshot_ui:.2f} days:**")
    st.markdown(f"* Target water level (elevation): **{target_water_level_final_ui:.2f} m** (Target drawdown: **{initial_water_table_elev_ui - target_water_level_final_ui:.2f} m** from initial WT of {initial_water_table_elev_ui:.2f}m)")
    
    all_points_meet_target = True
    results_text_list = []
    for point in observation_points_to_check_ui:
        s_at_point = calculate_superposition_unconfined(point['coords'][0], point['coords'][1], wells_data_ui, K_conductivity_ui, b_initial_saturated_thickness_ui, Sy_specific_yield_ui, t_pumping_snapshot_ui, initial_water_table_elev_ui, borehole_diameter_ui, True)
        final_wt_at_point = initial_water_table_elev_ui - s_at_point
        point_text = f"- At **{point['name']}** ({point['coords'][0]:.1f}, {point['coords'][1]:.1f}m): Drawdown = **{s_at_point:.2f}m**, Final Water Table Elev. = **{final_wt_at_point:.2f}m**"
        if final_wt_at_point > target_water_level_final_ui:
            all_points_meet_target = False
            point_text += f" -- <font color='red'>TARGET NOT MET!</font>"
        else:
            point_text += f" -- <font color='green'>Target met.</font>"
        results_text_list.append(point_text)
    st.markdown("\n".join(results_text_list), unsafe_allow_html=True)
    
    if all_points_meet_target: 
        st.success("SUCCESS: All specified observation points meet the target water level at the snapshot time.")
    else: 
        st.warning("DESIGN ADJUSTMENT MAY BE NEEDED: Not all critical observation points meet the target water level. Consider increasing pumping rates, number of wells, adjusting well locations/depths, or allowing more time.")

    st.subheader("Time to Target Calculation (at Excavation Center)")
    target_drawdown_magnitude_ui = initial_water_table_elev_ui - target_water_level_final_ui
    
    if observation_points_to_check_ui and wells_data_ui: 
        obs_point_for_display_time_calc = observation_points_to_check_ui[0] 
        st.markdown(f"Estimating time to reach target water table ({target_water_level_final_ui:.2f} m elev.) at **{obs_point_for_display_time_calc['name']}** ({obs_point_for_display_time_calc['coords'][0]:.1f}, {obs_point_for_display_time_calc['coords'][1]:.1f}m):")
        
        condition_for_final_time_calc = (target_drawdown_magnitude_ui > 0) or \
                                        (target_drawdown_magnitude_ui == 0 and target_water_level_final_ui == initial_water_table_elev_ui)

        if condition_for_final_time_calc:
            if target_drawdown_magnitude_ui == 0 and target_water_level_final_ui == initial_water_table_elev_ui:
                st.info("Target drawdown is 0m (target level is initial water table). Time to reach is effectively 0 days.")
            else: 
                with st.spinner("Calculating time to target for display..."):
                    time_to_target_display = find_time_to_target_drawdown_unconfined(
                        obs_point_for_display_time_calc['coords'][0], obs_point_for_display_time_calc['coords'][1], 
                        target_drawdown_magnitude_ui, wells_data_ui, 
                        K_conductivity_ui, b_initial_saturated_thickness_ui, Sy_specific_yield_ui, 
                        initial_water_table_elev_ui, borehole_diameter_ui)
                
                if isinstance(time_to_target_display, str): 
                    st.info(time_to_target_display)
                else: 
                    st.success(f"Estimated time to reach target water table level at {obs_point_for_display_time_calc['name']}: **{time_to_target_display:.3f} days**.")
        else: 
             st.warning(f"Target drawdown ({target_drawdown_magnitude_ui:.2f}m) is not positive. Ensure target level is below initial water table for a meaningful time calculation.")
    elif not wells_data_ui:
        st.warning("No wells defined, cannot calculate time to target.")
    else:
        st.warning("No observation points defined for time-to-target calculation display.")

if __name__ == "__main__":
    main()
