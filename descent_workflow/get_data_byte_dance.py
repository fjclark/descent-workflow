"""Functionality for getting the ByteDance torsion data, particularly guessing the angles and
dihedral indices, which are not provided in the dataset.
"""

import json
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlretrieve

import h5py
import MDAnalysis as mda
import numpy as np
from loguru import logger
from MDAnalysis.analysis.dihedrals import Dihedral
from numpy import typing as npt
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule
from openff.units import unit
from tqdm import tqdm
from yammbs.torsion.inputs import QCArchiveTorsionDataset, QCArchiveTorsionProfile

SAGE = ForceField("openff-2.2.0.offxml")
FFS_TO_TEST = [
    "openff_unconstrained-1.0.0.offxml",
    "/home/campus.ncl.ac.uk/nfc78/research/smee-spice-valence/descent-workflow/workflow/output_ff/spice2_linearised_harmonics_minibatch_systematic_tor_types_2_2_levels_fixed.offxml",
    "/home/campus.ncl.ac.uk/nfc78/research/smee-spice-valence/descent-workflow/workflow/output_ff/spice2_linearised_harmonics_minibatch_systematic_tor_types_ring_membership.offxml",
]

SPACING_MULTIPLES = {
    "BDTorsionNonRing": 15.0,  # Always spaced by 15 degrees
    "BDTorsionInRing": 5.0,
}  # Mostly spaced by 10 degrees, but some by 5 degrees

CHECK_ABSOLUTE = {
    "BDTorsionNonRing": True,
    "BDTorsionInRing": False,
}

H5_FILES = {
    "BDTorsionInRing": "BDTorsionInRing.h5",
    "BDTorsionNonRing": "BDTorsionNonRing.h5",
}

JSON_FILES = {
    "BDTorsionInRing": "BDTorsionInRing.json",
    "BDTorsionNonRing": "BDTorsionNonRing.json",
}

DATA_URLS = {
    "BDTorsionInRing.json": "http://raw.githubusercontent.com/bytedance/byteff/refs/heads/master/data/BDTorsionInRing.json",
    "BDTorsionInRing.h5": "https://github.com/bytedance/byteff/raw/refs/heads/master/data/BDTorsionInRing.h5",
    "BDTorsionNonRing.json": "https://raw.githubusercontent.com/bytedance/byteff/refs/heads/master/data/BDTorsionNonRing.json",
    "BDTorsionNonRing.h5": "https://github.com/bytedance/byteff/raw/refs/heads/master/data/BDTorsionNonRing.h5",
}
"""
Note on the data:
This folder contains data for BDTorsion. The two subsets (InRing and NonRing) are saved seperated in JSON and H5 files, respectively.
The JSON files contain the mapping between molecule names and mapped SMILES. Mapped SMILES can be parsed by RDKit to reconstruct molecular graphs.
The data in the H5 files are grouped by molecule names. Within each group, there are three datasets: coords, forces and energy. The shape of coords and forces is [# conformers, # atoms, 3], and the shape of energy is [# conformers].
"""


def download_raw_data(
    data_dir: Path, dataset_name: Literal["BDTorsionInRing", "BDTorsionNonRing"]
) -> dict[str, Path]:
    """Download the raw data from the ByteDance torsion dataset.

    Args:
        data_dir (Path): The directory to download the data to.
        dataset_name (str): The name of the dataset to download. Either "BDTorsionInRing" or "BDTorsionNonRing".

    Returns:
        dict[str, Path]: A dictionary mapping file names to their local paths.

    """
    data_dir.mkdir(parents=True, exist_ok=True)
    local_paths = {}
    desired_files = [H5_FILES[dataset_name], JSON_FILES[dataset_name]]

    for file_name in desired_files:
        url = DATA_URLS[file_name]
        local_path = data_dir / file_name
        if not local_path.exists():
            logger.info(f"Downloading {file_name} from {url} to {local_path}")
            urlretrieve(url, local_path)
        else:
            logger.info(f"File {local_path} already exists, skipping download.")
        local_paths[file_name] = local_path

    return local_paths


def get_mols_with_conformers_and_energies(
    h5file: Path, jsonfile: Path
) -> tuple[dict[str, Molecule], dict[str, Any]]:
    """Load molecules and their conformers from H5 and JSON files.

    Args:
        h5file (Path): Path to the H5 file containing conformer data.
        jsonfile (Path): Path to the JSON file containing SMILES data.

    Returns:
        tuple: Dictionary mapping molecule names to Molecule objects with conformers,
               and dictionary mapping molecule names to their energies.

    """
    with open(jsonfile, "r") as f:
        smiles_data = json.load(f)

    mols = {}
    all_energies = {}

    with h5py.File(h5file, "r") as h5f:
        for mol_name, smiles in smiles_data.items():
            group = h5f[mol_name]
            coords = [c * unit.angstrom for c in group["coords"][:]]  # Convert to OpenFF units
            energies = group["energy"][:] * unit.kilocalorie_per_mole  # Convert to OpenFF units
            # Subtract minimum energy to get relative energies
            energies -= np.min(energies)

            off_mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
            off_mol._conformers = list(coords)

            mols[mol_name] = off_mol
            all_energies[mol_name] = energies

    return mols, all_energies


def get_dihedral_angles(mol: Molecule, atom_indices: list[list[int]]) -> list[list[float]]:
    """Calculate dihedral angles for a list of conformers given the atom indices.

    Args:
        mol (Molecule): The molecule object.
        confs (list[np.ndarray]): List of conformer coordinates.
        atom_indices list[list[int]]: List of lists of four atom indices defining the dihedrals.

    Returns:
        list[list[float]]: List of lists of dihedral angles for each set of atoms.

    """
    confs = mol.conformers

    u = mda.Universe.empty(n_atoms=mol.n_atoms, trajectory=True)
    u.add_TopologyAttr("elements", [atom.atomic_number for atom in mol.atoms])
    u.load_new(np.array(confs), order="fac")
    atom_groups = [u.atoms[list(indices)] for indices in atom_indices]
    angles_list = Dihedral(atom_groups).run().angles
    # Convert values < 0 to above 180
    for angles in angles_list:
        angles = [angle + 360 if angle < 0 else angle for angle in angles]

    # Rearrange the angles list so that it has shape (n_dihedrals, n_conformers)
    angles_list_transposed: list[list[float]] = np.array(angles_list).T.tolist()
    return angles_list_transposed


def get_angle_periodic_diff(angle1: float, angle2: float) -> float:
    """Calculate the smallest difference between two angles, accounting for periodicity.

    Args:
        angle1 (float): First angle in degrees.
        angle2 (float): Second angle in degrees.

    Returns:
        float: Smallest difference between the two angles in degrees.

    """
    diff = angle2 - angle1
    diff_wrapped: float = float((diff + 180) % 360 - 180)  # Wrap to [-180, 180]
    return diff_wrapped


def are_multiples_of_spacing(
    angles: list[float] | npt.NDArray[np.floating[Any]],
    spacing_multiple: float = 5.0,
    tolerance: float = 0.02,
    n_allowed_deviations: int = 0,
) -> bool:
    """Check if all angles are multiples of the given spacing within a tolerance."""
    is_close = np.isclose(
        [min(abs(angle % spacing_multiple), abs(angle % -spacing_multiple)) for angle in angles],
        0,
        atol=tolerance,
    )

    result: bool = bool(np.sum(~is_close) <= n_allowed_deviations)
    return result


def is_torsion_scan(
    angles: list[float],
    spacing_multiple: float = 5.0,
    tolerance: float = 0.02,
    check_absolute: bool = False,
) -> bool:
    """Check if a list of angles represents a torsion scan with given spacing and tolerance.
    Account for cases where some angles may be missing.

    Args:
        angles (list[float]): List of dihedral angles.
        spacing_multiple (float, optional): The expected spacing should be a multiple of
        this value (in degrees). Defaults to 5.0.
        tolerance (float, optional): Tolerance for spacing check. Defaults to 0.02.
        check_absolute (bool, optional): If True, check that all angles (not just differences)
        are multiples of spacing_multiple within tolerance. Defaults to False.

    Returns:
        bool: True if the angles represent a torsion scan, False otherwise.

    """
    angles = sorted(angles)

    # Check that the angles all differ by at least spacing_multiple
    if max(angles) - min(angles) < spacing_multiple - tolerance:
        return False

    # Check that all angles divide evenly by the spacing
    if check_absolute:
        if not are_multiples_of_spacing(
            angles, spacing_multiple=spacing_multiple, tolerance=tolerance
        ):
            return False

    angle_diffs = np.array(
        [get_angle_periodic_diff(i, j) for i, j in zip(angles[:-1], angles[1:], strict=True)]
    )

    return are_multiples_of_spacing(
        angle_diffs,
        spacing_multiple=spacing_multiple,
        tolerance=tolerance,
    )


def get_dihedral_indices_and_angles(
    mol: Molecule,
    name: str,
    spacing_multiple: float = 5.0,
    check_absolute: bool = False,
) -> tuple[tuple[int, int, int, int], list[float]]:
    """For all bonds, get the set of dihedral angles. Pick the set of angles most consistent
    with being a torsion scan (spaced by 15 degrees) and return the corresponding atom indices
    and angles.

    Args:
        mol (Molecule): The molecule to analyse.
        name (str): The name of the molecule (for logging).
        spacing_multiple (float, optional): The expected spacing should be a multiple of
        this value (in degrees). Defaults to 5.0.
        check_absolute (bool, optional): If True, check that all angles (not just differences)
        are multiples of spacing_multiple within tolerance. Defaults to False.

    Returns:
        tuple[tuple[int, int, int, int], list[float]]: The dihedral atom indices and angles.

    """
    indices_list = []
    # Get the set of all possible dihedral atoms
    top = mol.to_topology()
    for p in top.propers:  # None appear to be impropers in this dataset
        indices_list.append([a.molecule_atom_index for a in p])

    angles_list = get_dihedral_angles(mol, indices_list)
    angles_by_indices = {
        tuple(indices): angle for indices, angle in zip(indices_list, angles_list, strict=True)
    }

    # Now, find all sets of indices where the angles are spaced by ~15 degrees.
    torsion_scans = {
        indices: angles
        for indices, angles in angles_by_indices.items()
        if is_torsion_scan(angles, spacing_multiple=spacing_multiple, check_absolute=check_absolute)
    }

    if not torsion_scans:
        raise ValueError(f"No torsion scan found for molecule {name}.")

    if len(torsion_scans) > 1:
        raise ValueError(
            f"Multiple torsion scans ({len(torsion_scans)}) found for molecule {name}: "
            f"{torsion_scans}."
        )

    torsion_item = list(torsion_scans.items())[0]
    return torsion_item


def can_parameterise(mol: Molecule, ff_names: list[str]) -> bool:
    """Check if a molecule can be parameterised with all the given force fields.

    Args:
        mol (Molecule): The molecule to check.
        ff_names (list[str]): The names of the force fields to check.

    Returns:
        bool: True if the molecule can be parameterised, False otherwise.

    """
    for ff_name in ff_names:
        ff = ForceField(ff_name)
        try:
            Interchange.from_smirnoff(ff, [mol])
        except Exception as e:
            logger.warning(f"Could not parameterise molecule {mol.to_smiles()} with {ff_name}: {e}")
            return False

    return True


def create_qca_torsion_dataset(
    indices_and_angles: dict[str, tuple[tuple[int, int, int, int], list[float]]],
    mols: dict[str, Molecule],
    all_energies: dict[str, Any],
) -> QCArchiveTorsionDataset:
    """Create a QCArchiveTorsionDataset from torsion scan data and save to a JSON file.

    Args:
        indices_and_angles (dict): Dictionary mapping molecule names to tuples of
        dihedral indices and angles.
        mols (dict): Dictionary mapping molecule names to Molecule objects with conformers.
        all_energies (dict): Dictionary mapping molecule names to their energies.

    Returns:
        QCArchiveTorsionDataset: The constructed torsion dataset.

    """
    qm_torsions = []

    for name, (indices, angles) in tqdm(indices_and_angles.items()):
        idx = name.split("/")[-1]
        mol = mols[name]

        if not can_parameterise(mol, FFS_TO_TEST):
            logger.warning(f"Skipping molecule {name} as it cannot be parameterised.")
            continue

        mapped_smiles = mol.to_smiles(mapped=True)

        coords = np.array([conf.m_as(unit.angstrom) for conf in mol.conformers])
        coords_dict = dict(zip(angles, coords, strict=True))
        energies = all_energies[name].m_as(unit.kilocalorie_per_mole)
        energies_dict = dict(zip(angles, energies, strict=True))

        qm_torsions.append(
            QCArchiveTorsionProfile(
                id=idx,
                mapped_smiles=mapped_smiles,
                dihedral_indices=indices,
                coordinates=coords_dict,
                energies=energies_dict,
            )
        )

    logger.info(f"Created {len(qm_torsions)} torsion profiles from {len(mols)} molecules.")

    return QCArchiveTorsionDataset(qm_torsions=qm_torsions)


def get_data_byte_dance(
    output_dir: str, dataset_name: Literal["BDTorsionInRing", "BDTorsionNonRing"]
) -> None:
    """Download the ByteDance torsion dataset, process it to find dihedral indices and angles.

    Args:
        output_dir (str): Directory to download and store the data.
        dataset_name (str): Name of the dataset to download.
            Either "BDTorsionInRing" or "BDTorsionNonRing".

    """
    output_dir_path = Path(output_dir)
    logger.info(f"Getting ByteDance torsion data for dataset {dataset_name}")
    local_paths = download_raw_data(output_dir_path, dataset_name)
    mols, all_energies = get_mols_with_conformers_and_energies(
        local_paths[H5_FILES[dataset_name]], local_paths[JSON_FILES[dataset_name]]
    )

    spacing_multiple = SPACING_MULTIPLES[dataset_name]
    check_absolute = CHECK_ABSOLUTE[dataset_name]
    indices_and_angles = {
        name: get_dihedral_indices_and_angles(
            mol, name, spacing_multiple=spacing_multiple, check_absolute=check_absolute
        )
        for name, mol in tqdm(mols.items())
    }

    torsion_dataset = create_qca_torsion_dataset(indices_and_angles, mols, all_energies)

    output_path = output_dir_path / "qca-torsion-data.json"
    with open(output_path, "w") as f:
        f.write(torsion_dataset.json())

    logger.info(f"Saved torsion dataset to {output_path}")
