// ==============================================================================
// This file is part of THOR.
//
//     THOR is free software : you can redistribute it and / or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
//
//     THOR is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//     GNU General Public License for more details.
//
//     You find a copy of the GNU General Public License in the main
//     THOR directory under <license.txt>.If not, see
//     <http://www.gnu.org/licenses/>.
// ==============================================================================
//
//
//
//
// Description: Insolation computation
//
//
//
// Known limitations: None
//
// Known issues: None.
//
//
// If you use this code please cite the following reference:
//
//       [1] Mendonca, J.M., Grimm, S.L., Grosheintz, L., & Heng, K., ApJ, 829, 115, 2016
//
// Current Code Owners: Joao Mendonca (joao.mendonca@space.dtu.dk)
//                      Russell Deitrick (russell.deitrick@csh.unibe.ch)
//                      Urs Schroffenegger (urs.schroffenegger@csh.unibe.ch)
//
// History:
// Version Date       Comment
// ======= ====       =======
// 2.0     30/11/2018 Released version (RD & US)
// 1.0     16/08/2017 Released version  (JM)
//
////////////////////////////////////////////////////////////////////////
#pragma once

#include "cuda_device_memory.h"

#include "config_file.h"
#include "define.h"
#include "dyn/phy_modules_device.h"
#include "log_writer.h"
#include "simulation_setup.h"
#include "storage.h"

#define insol_avg_default "NoInsolAvg"
enum insolation_average_types { NO_INSOL_AVG = 0, DIURNAL_AVG = 1, ANNUAL_AVG = 2 };

class ESP;

class Insolation
{
public:
    Insolation();

    // Called by any module to enable insolation computation
    void set_require() {
        enabled = true;
    };

    bool configure(config_file &config_reader);

    void print_config();

    bool initialise_memory(const ESP &esp, device_RK_array_manager &phy_modules_core_arrays);
    bool initial_conditions(const ESP &esp, const SimulationSetup &sim, storage *s);

    bool store_init(storage &s);

    bool phy_loop(ESP &                  esp,
                  const SimulationSetup &sim,
                  int                    nstep, // Step number
                  double                 time_step);            // Time-step [s]

    bool store(const ESP &esp, storage &s);

    // get pointer to data on device
    double *get_device_cos_zenith_angles() {
        return *cos_zenith_angles;
    };

    // fetch data to host and get pointer to data
    std::shared_ptr<double[]> get_host_cos_zenith_angles() {
        return cos_zenith_angles.get_host_data();
    };

    double get_r_orb() {
        return r_orb;
    };

    double get_r_orb_host() {
        return r_orb_host;
    };

    double get_mean_motion() {
        return mean_motion;
    };
    
        // get pointer to data on device
    double *get_device_cos_zenith_angles_moon() {
        return *cos_zenith_angles_moon;
    };

    // fetch data to host and get pointer to data
    std::shared_ptr<double[]> get_host_cos_zenith_angles_moon() {
        return cos_zenith_angles_moon.get_host_data();
    }

    bool get_eclipse_status() {
        return eclipse_status;
    };

    double get_Fraction_reflection() {
        return Fraction_reflection;
    };

    double get_moon_orbit_distance_change() {
        return moon_orbit_distance_change;
    };

        // get pointer to data on device
    double *get_device_cos_zenith_angles_S1() {
        return *cos_zenith_angles_S1;
    };

        // get pointer to data on device
    double *get_device_cos_zenith_angles_S2() {
        return *cos_zenith_angles_S2;
    };

    double get_shadow_F1() {
        return shadow_F1;
    };

    double get_shadow_F2() {
        return shadow_F2;
    };

    double get_a_p1() {
        return a_p1;
    };

    double get_a_p2() {
        return a_p1;
    };

    double get_incflx_S1() {
        return incflx_final_S1;
    };

    double get_incflx_S2() {
        return incflx_final_S2;
    };

    double get_incflx_IR() {
        return incflx_IR;
    };

    double get_incflx_reflection() {
        return incflx_reflection;
    };
    

private:
    bool enabled = false;


    // insolation computation vars from rt module
    // orbit/insolation properties
    bool                       sync_rot       = true;     // is planet syncronously rotating?
    double                     mean_motion    = 1.991e-7; // orbital mean motion (rad/s)
    double                     mean_anomaly_i = 0;        // initial mean anomaly at start (rad)
    double                     mean_anomaly   = 0;        // current mean anomaly of planet (rad)
    double                     true_long_i    = 0;        // initial true longitude of planet (rad)
    double                     ecc            = 0;        // orbital eccentricity
    double                     obliquity      = 0;        // obliquity (tilt of spin axis) (rad)
    double                     r_orb          = 1;        // orbital distance/semi-major axis
    double                     sin_decl       = 0; // declination of host star (relative to equator)
    double                     cos_decl       = 1;
    double                     alpha_i = 0; // initial right asc of host star (relative to long = 0)
    double                     alpha   = 0; // right asc of host star (relative to long = 0)
    double                     longp   = 0; // longitude of periastron (rad)
    double                     Pday;        //rotation period (s)
    double                     Porb;        //orbital period (s)
    int                        n_days_orb;  //number of days per orbit
    cuda_device_memory<double> cos_zenith_daily; //daily averaged zenith angles
    cuda_device_memory<double> day_start_time; //start time of each day (relative to start of orbit)
    int                        n_day_steps = 100; //num of steps to do daily average over

    bool   sync_rot_config    = true;     // is planet syncronously rotating?
    double mean_motion_config = 1.991e-7; // orbital mean motion (rad/s)
    double true_long_i_config = 0;        // initial true longitude of planet (rad)
    double ecc_config         = 0;        // orbital eccentricity
    double obliquity_config   = 0;        // obliquity (tilt of spin axis) (rad)
    double alpha_i_config     = 0;        // initial right asc of host star (relative to long = 0)
    double longp_config       = 0;        // longitude of periastron (rad)
    
    // moon mod
    bool   moon_irr_config    = false;    // simulated moon irradiated by host planet
    double moon_host_D_config = 0;        // distance between moon and host planet
    double radius_host_config = 0;        // radius of the host planet
    bool   moon_irr_mode      = false;    // simulated moon irradiated by host planet
    double moon_host_D        = 0;        // distance between moon and host planet
    double radius_host        = 0;        // radius of the host planet
    double moon_irr           = 0;        //    
    double mean_motion_host_config = 1.991e-7; // orbital mean motion of host planet (rad/s)    
    double mean_motion_host        = 1.991e-7; // orbital mean motion of host planet (rad/s)    
    double mean_anomaly_host_i     = 0;        // initial mean anomaly of host planet at start (rad)
    double mean_anomaly_host       = 0;        // current mean anomaly of host planet (rad)    
    double ecc_host                = 0;        // orbital eccentricity of host planet    
    double ecc_host_config         = 0;        // orbital eccentricity of host planet    
    double r_orb_host              = 1;        // orbital distance/semi-major axis
    //double dayside_contribution  = 0;        // fraction of fluxes coming from the dayside    
    //double nightside_contribution= 0;        // fraction of fluxes coming from the nightside
    double Fraction_reflection     = 0;   // fraction of fluxes coming from the dayside
    bool   eclipse_status          = false; 
    double moon_orbit_distance_change = 0.0;  // relative changes due to moon orbit

    bool   binary_star_mode_config      = false;    // mode to simulate binary star systems
    double Tstar_primary                = 0.0;      // stellar temperature (k) of primary star
    double radius_star_primary          = 0.0;      // radius of host star (R_sun) of the primary star
    double Tstar_secondary              = 0.0;      // stellar temperature (k) of secondary star
    double radius_star_secondary        = 0.0;      // radius of host star (R_sun) of the secondary star
    double a_secondary                  = 0.0;      // Semimajor axis of the secondary star (AU) around the barycenter
    double a_primary                    = 0.0;      // Semimajor axis of the primary star (AU) around the barycenter
    double binary_perdiod               = 0.0;      // Orbital period of the binary stars (days)
    double M_S1                         = 0.0;      // Mass of the primary star (M_Sun)
    double M_S2                         = 0.0;      // Mass of the secondary star (M_Sun)
    double Tstar_secondary_config       = 0.0;      // stellar temperature (k) of secondary star
    double radius_star_secondary_config = 0.0;      // radius of host star (R_sun) of the secondary star
    double Tstar_primary_config         = 0.0;      // stellar temperature (k) of secondary star
    double radius_star_primary_config   = 0.0;      // radius of host star (R_sun) of the secondary star
    double a_secondary_config           = 0.0;      // Semimajor axis of the secondary star (AU) around the barycenter
    double a_primary_config             = 0.0;      // Semimajor axis of the primary star (AU) around the barycenter
    double binary_perdiod_config        = 0.0;      // Orbital period of the binary stars (days)
    double M_S1_config                  = 0.0;      // Mass of the primary star (M_Sun)
    double M_S2_config                  = 0.0;      // Mass of the secondary star (M_Sun)
    double alpha_moon_C                 = 0.0;
    double alpha_S1                     = 0.0;
    double alpha_S2                     = 0.0;
    double alpha_day                    = 0.0;
    double omega_S1                     = 0.0;
    double omega_day                    = 0.0;
    double omega_moon_orbit_Fday        = 0.0;
    double moon_orbit_F_rec             = 0.0;
    double planet_star_dist             = 0.0;
    double a_pc                         = 0.0;
    double a_S2                         = 0.0;
    double a_S1                         = 0.0;
    double gamma_S1                     = 0.0;
    double gamma_S2                     = 0.0;
    double a_p1                         = 0.0;
    double a_p2                         = 0.0;
    double apparent_R_S1                = 0.0;    
    double apparent_R_S2                = 0.0;
    double shadow_F1                    = 0.0;
    double shadow_F2                    = 0.0;
    double phi_S1                       = 0.0;
    double phi_S2                       = 0.0;
    double D_critical                   = 0.0;    
    double d1                           = 0.0;
    double d2                           = 0.0;
    double A_intersection               = 0.0;
    double incflx_final_S1              = 0.0;
    double incflx_final_S2              = 0.0;
    double incflx_IR                    = 0.0;
    double incflx_reflection            = 0.0;

    bool print_once            = true;
    
    double table_num_parmentier_config = 2; // if picket-fence, table number used to compute gamma (with/without Tio)

    insolation_average_types insol_avg; // use averaging of insolation/stellar forcing
    string                   insol_avg_str;

    cuda_device_memory<double> cos_zenith_angles;
    cuda_device_memory<double> cos_zenith_angles_moon;
    cuda_device_memory<double> cos_zenith_angles_S1;
    cuda_device_memory<double> cos_zenith_angles_S2;

    void update_spin_orbit(double time, double Omega, bool moon_irr_mode, bool binary_star_mode);
};
