import numpy as np
from gfalm.main import run
from gfalm.utils import read_input_file

print('Modules imported!')


def energy_volume(input_file: str, scales: np.ndarray, energy_output='PBE', return_mom=False):
    """Calculates the energies for a set of different volumes. Input for EOS-fitting.

    Parameters
    ----------
    input_file : str
        Green Alm specific input file (".in") for the calculation.
    scales : np.ndarray
        Scales for the lattice parameters.
    energy_output : string
        Either PBE or PBEsol. Default is PBE.
    return_mom : bool, optional
        A flag used to indicate if magnetic moments should be
        returned or not. Default is false.

    Returns
    -------
    Tuple
        If return_mom=True then energies and magnetic moments will be returned else only energies. 
    """
    stored_energies = []  # list, where the results will be stored
    input_data = read_input_file(input_file)
    # Remember the initial job name
    job_name = input_data['ctrl_input']['job_name']

    if return_mom == True:
        mag_moms = []  # list, where the magnetic moments will be stored

    # Run a loop over sws parameters (provided as function input)
    for scale in scales:

        # We modify the lattice scale. Note how the units are specified.
        input_data['struct_input']['lattice_scale'] = (scale, 'A')

        # Add a prefix to the initial job_name to identify each calculation
        input_data['ctrl_input']['job_name'] = "%s_a%.2f" % (job_name, scale)

        # Run GreenALM library main routine
        result = run(**input_data)

        # Extract the total energy, either PBE or PBEsol
        general = result['general']
        if energy_output == 'PBE':
            tot_energy = general.total_energy_xc2
        else:
            tot_energy = general.total_energy_xc3

        # Add the obtained energy to the list
        stored_energies.append(tot_energy)

        if return_mom == True:
            latt_site = result['sites'][0]
            mom = latt_site.mag_mom[0]
            mag_moms.append(mom)

    # Return the results
    if return_mom == True:
        return np.array(stored_energies), np.array(mag_moms)
    else:
        return np.array(stored_energies)


print("Function defined successfully")
