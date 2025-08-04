import pyvista as pv
import numpy as np
import os
import xml.etree.ElementTree as ET # For robust .pvd parsing

# Import Firedrake components needed for saving data as VTKFile
from firedrake import Function, FunctionSpace, Mesh
from firedrake.output import VTKFile

def get_member_data_at_timestep(pvd_filepath, timestep_idx, member_id):
    """
    Loads 1D spatial data for a single ensemble member at a specific timestep.

    Args:
        pvd_filepath (str): Path to the .pvd file for the specific ensemble member.
        timestep_idx (int): The index of the timestep to load (0-based).
        member_id (int): The ID of the ensemble member, used to construct the
                         expected data array name in the VTK file.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: The 1D spatial data for this member at this timestep.
                             Returns None if data cannot be loaded or found.
            - float: The actual timestep value from the PVD file. Returns None if not found.
    """
    try:
        # Parse the PVD file to get individual VTK file paths and timesteps
        tree = ET.parse(pvd_filepath)
        root = tree.getroot()
        collection = root.find('Collection')

        if collection is None:
            print(f"Warning: No 'Collection' tag found in {pvd_filepath}.")
            return None, None

        # Get the directory of the PVD file to construct full paths
        pvd_dir = os.path.dirname(pvd_filepath)

        # Sort datasets by timestep to ensure correct indexing
        datasets = sorted(collection.findall('DataSet'), key=lambda x: float(x.get('timestep')))

        if timestep_idx >= len(datasets):
            print(f"Warning: Timestep index {timestep_idx} out of range for {pvd_filepath}.")
            return None, None

        dataset_elem = datasets[timestep_idx]
        timestep_value = float(dataset_elem.get('timestep'))
        relative_filepath = dataset_elem.get('file')
        vtu_filepath = os.path.join(pvd_dir, relative_filepath)

        # Use pyvista to read the VTK file
        mesh = pv.read(vtu_filepath)

        # Construct the data array name based on the member_id
        # This matches how you named your Firedrake Functions: f"particle_{p}"
        data_array_name_in_vtu = f"particle_{member_id}"

        if data_array_name_in_vtu in mesh.point_data:
            spatial_data = mesh.point_data[data_array_name_in_vtu].flatten()
        elif data_array_name_in_vtu in mesh.cell_data:
            spatial_data = mesh.cell_data[data_array_name_in_vtu].flatten()
        else:
            # Fallback for older pyvista versions or different naming conventions
            # Sometimes, if only one scalar array exists, it might be the default active one.
            if mesh.active_scalars is not None:
                spatial_data = mesh.active_scalars.flatten()
                print(f"Info: Using active_scalars for {vtu_filepath} as '{data_array_name_in_vtu}' not found directly.")
            else:
                print(f"Warning: Data array '{data_array_name_in_vtu}' not found in {vtu_filepath} and no active scalars. Skipping.")
                return None, None

        if len(spatial_data) == 0:
            print(f"Warning: No data found for array '{data_array_name_in_vtu}' in {vtu_filepath}.")
            return None, None

        return spatial_data, timestep_value

    except Exception as e:
        print(f"Error loading data for member {member_id} at timestep index {timestep_idx} from {pvd_filepath}: {e}")
        return None, None


def calculate_ensemble_statistics(ensemble_pvd_filepaths, num_timesteps, num_ensemble_members,
                                  output_dir, firedrake_mesh, firedrake_function_space):
    """
    Calculates ensemble mean and variance for each (spatial point, timestep)
    and saves the ensemble mean and variance to new PVD files.

    Args:
        ensemble_pvd_filepaths (list): A list of paths to the .pvd file for each
                                       ensemble member (e.g., ['path/to/particle_0.pvd', ...]).
        num_timesteps (int): The total number of timesteps to process.
        num_ensemble_members (int): The total number of ensemble members.
        output_dir (str): Directory where the 'particle_mean.pvd' and 'particle_var.pvd'
                          files will be saved.
        firedrake_mesh (firedrake.Mesh): The Firedrake mesh object used for the simulation.
        firedrake_function_space (firedrake.FunctionSpace): The Firedrake FunctionSpace
                                                            used for the data (e.g., V).

    Returns:
        tuple:
            - ensemble_means_over_time (dict): {timestep_value: 1D numpy array of ensemble means}
            - ensemble_variances_over_time (dict): {timestep_value: 1D numpy array of ensemble variances}
    """
    ensemble_means_over_time = {}
    ensemble_variances_over_time = {}

    print(f"Starting ensemble statistics calculation for {num_ensemble_members} members over {num_timesteps} timesteps...")

    # Determine the number of spatial points from the first ensemble member's first timestep
    # This assumes all members and timesteps have the same spatial discretization.
    first_member_id = 0
    first_pvd_path = ensemble_pvd_filepaths[first_member_id]
    temp_data, _ = get_member_data_at_timestep(first_pvd_path, 0, first_member_id)
    if temp_data is None:
        print("Error: Could not determine spatial dimension from first ensemble member. Aborting.")
        return {}, {}
    num_spatial_points = len(temp_data)
    print(f"Detected {num_spatial_points} spatial points.")

    # Setup VTKFile for saving the ensemble mean
    mean_output_filepath = os.path.join(output_dir, "particle_mean.pvd")
    ensemble_mean_file = VTKFile(mean_output_filepath)
    f_mean = Function(firedrake_function_space, name="ensemble_mean")

    # Setup VTKFile for saving the ensemble variance
    var_output_filepath = os.path.join(output_dir, "particle_var.pvd")
    ensemble_var_file = VTKFile(var_output_filepath)
    f_var = Function(firedrake_function_space, name="ensemble_variance")

    for t_idx in range(num_timesteps):
        print(f"  Processing global timestep index {t_idx}...")
        all_member_data_at_current_timestep = []
        current_timestep_value = None

        for member_id in range(num_ensemble_members):
            member_pvd_path = ensemble_pvd_filepaths[member_id]
            spatial_data_for_member, ts_value = get_member_data_at_timestep(
                member_pvd_path, t_idx, member_id
            )

            if spatial_data_for_member is None:
                print(f"    Warning: Skipping member {member_id} at timestep index {t_idx} due to data loading issues.")
                continue

            if current_timestep_value is None:
                current_timestep_value = ts_value
            elif current_timestep_value != ts_value:
                print(f"    Warning: Timestep value mismatch for member {member_id} at index {t_idx}. Expected {current_timestep_value}, got {ts_value}. Using first detected.")

            if len(spatial_data_for_member) != num_spatial_points:
                print(f"    Warning: Spatial dimension mismatch for member {member_id} at timestep index {t_idx}. Expected {num_spatial_points}, got {len(spatial_data_for_member)}. Skipping member.")
                continue

            all_member_data_at_current_timestep.append(spatial_data_for_member)

        if not all_member_data_at_current_timestep:
            print(f"    No valid data loaded for any member at timestep index {t_idx}. Skipping this timestep.")
            continue

        # Convert list of 1D arrays to a 2D NumPy array
        # Shape will be (num_valid_members, num_spatial_points)
        ensemble_data_at_t = np.array(all_member_data_at_current_timestep)

        # Calculate ensemble mean and variance along the ensemble dimension (axis=0)
        # Resulting shape will be (num_spatial_points,)
        current_ensemble_mean = np.mean(ensemble_data_at_t, axis=0)
        current_ensemble_variance = np.var(ensemble_data_at_t, axis=0)

        if current_timestep_value is not None:
            ensemble_means_over_time[current_timestep_value] = current_ensemble_mean
            ensemble_variances_over_time[current_timestep_value] = current_ensemble_variance
            print(f"    Finished timestep {current_timestep_value}. Processed {ensemble_data_at_t.shape[0]} members.")

            # Save the ensemble mean to the PVD file
            f_mean.dat.data[:] = current_ensemble_mean
            ensemble_mean_file.write(f_mean, time=current_timestep_value)

            # Save the ensemble variance to the PVD file
            f_var.dat.data[:] = current_ensemble_variance
            ensemble_var_file.write(f_var, time=current_timestep_value)
        else:
            print(f"    Could not determine timestep value for index {t_idx}. Skipping storing results for this timestep.")

    print(f"\nEnsemble mean saved to: {mean_output_filepath}")
    print(f"Ensemble variance saved to: {var_output_filepath}")
    return ensemble_means_over_time, ensemble_variances_over_time

# --- Main execution block ---
if __name__ == "__main__":
    # IMPORTANT: These parameters must match your actual simulation output
    # Ensure Nensemble in your data generation script is set to 60 (or your desired number).
    num_ensemble_members = 60 # Set this to your actual number of ensemble members
    num_timesteps = 40 # Based on N_obs / ndump in your generation code (e.g., 16000 / 100 = 160)
                       # You need to know the actual number of timesteps written to each PVD.

    output_dir = "../../DA_Results/SCH/SFLT/SCH_particles/" # Must match your data generation path

    # --- Dummy Firedrake setup for demonstration ---
    # In your actual script, you would get these from your 'model' object
    # For this example, we'll create a dummy mesh and function space
    # to allow the script to run standalone for testing.
    # Replace this with your actual model.mesh and FunctionSpace(model.mesh, "CG", 1)
    # from your simulation setup.
    from firedrake import UnitIntervalMesh, FunctionSpace
    dummy_mesh = UnitIntervalMesh(5000) # Matches your xpoints=5000 for 1D
    dummy_V = FunctionSpace(dummy_mesh, "CG", 1)
    # ------------------------------------------------

    # Construct the list of .pvd file paths for all ensemble members
    ensemble_pvd_files = [os.path.join(output_dir, f"particle_{p}.pvd") for p in range(num_ensemble_members)]

    # Check if the PVD files exist before attempting to process
    missing_files = [f for f in ensemble_pvd_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: The following PVD files are missing. Please ensure your data generation script has run correctly with Nensemble={num_ensemble_members} and the output directory is correct:")
        for mf in missing_files:
            print(f"- {mf}")
        print("\nExiting. Please generate the data first.")
    else:
        # Calculate ensemble statistics and save the mean
        ensemble_means, ensemble_variances = calculate_ensemble_statistics(
            ensemble_pvd_files, num_timesteps, num_ensemble_members,
            output_dir, dummy_mesh, dummy_V # Pass the Firedrake mesh and function space
        )

        print("\n--- Summary of Ensemble Mean and Variance per Timestep ---")
        if ensemble_means:
            for ts, mean_data in ensemble_means.items():
                print(f"Timestep {ts}:")
                print(f"  Ensemble Mean (first 5 spatial points): {mean_data[:5]}")
                print(f"  Ensemble Variance (first 5 spatial points): {ensemble_variances[ts][:5]}")
                print(f"  Shape of Mean/Variance at this timestep: {mean_data.shape}")
        else:
            print("No ensemble statistics could be calculated. Check warnings/errors above.")
