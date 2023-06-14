
import re
import os
import io
import bz2
import tqdm
import math
import gzip
import lzma
import yaml
import json
import click
import pickle
import warnings
import itertools
import numpy as np
import pandas as pd
from typing import *
from .utils import ropen
from operator import itemgetter
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

Vector = Tuple[float, float, float]
Record = Dict[str, Any]


def _dumps_yaml(data: Dict[Any, Any]) -> bytes:
    buffer = io.StringIO()
    yaml.safe_dump(data, stream=buffer)
    buffer.seek(0)
    return buffer.read().encode("utf-8")


REQUIRED_FILES: Set[str] = {"OSZICAR", "OUTCAR", "CONTCAR"}
FORCE_MARK = re.compile(r"^\s*POSITION\s+TOTAL-FORCE\s*\(eV\/Angst\)")
FORCE_REGEX = re.compile("".join(rf"\s*(?P<{name}>-?\d+\.\d+)" for name in ('x', 'y', 'z', "fx", "fy", "fz")))
FIELDS = ("N", "E", "dE", "d\s+eps", "ncg", "rms", "rms\(c\)")
IONIC_STEP_MARK = re.compile(r"^\s+" + r"\s+".join(FIELDS))
SCF_STEP_REGEX = re.compile(
    r"(?P<algo>DAV|RMM):\s+(?P<step>\d+)\s+(?P<e>-?\d*\.\d+E(\+|-)\d+)\s+(?P<de>-?\d*\.\d+E(\+|-)\d+)\s+(?P<deps>-?\d*\.\d+E(\+|-)\d{2})\s*(?P<ncg>\d+)\s+(?P<rms>-?\d*\.\d+E(\+|-)\d+)\s*(?P<rmsc>-?\d*\.\d+E(\+|-)\d+)?")
IONIC_STEP_REGEX = re.compile(
    r"\s*(?P<step>\d+)\s+F\=\s*(?P<f>-?\d*\.\d+E?(\+|-)?\d*)\s+E0\=\s*(?P<ezero>-?\d*\.\d+E?(\+|-)?\d*)\s+d\s*E\s*\=(?P<de>-?\d*\.\d+E?(\+|-)?\d*)\s*(mag\=)?\s*(?P<mag>-?\d*\.\d+E?(\+|-)?\d*)?")

COMPRESSIONS = {
    None: open,
    "gz": gzip.open,
    "xz": lzma.open,
    "bz2": bz2.BZ2File
}

REPRESENTERS = dict(
    pickle=pickle.dumps,
    json=lambda o: json.dumps(o).encode("utf-8"),
    yaml=_dumps_yaml
)

LOADERS = dict(
    pickle=pickle.loads,
    json=lambda o: json.loads(o.decode("utf-8")),
    yaml=lambda o: yaml.safe_load(o.decode("utf-8"))
)


def identity(f: Any) -> Any:
    return f


def star(f):
    return lambda x: f(*x)


def isa(clsz: Union[Type, Tuple[Type, ...]]) -> bool:
    return lambda o: isinstance(o, clsz)


def transpose(it: Iterable[Iterable[Any]]) -> Iterable[Iterable[Any]]:
    return zip(*it)


def is_vasp_calc_dir(folder: str, _: List[str], files: List[str]) -> bool:
    required_files_exist = all(req in files for req in REQUIRED_FILES)
    if required_files_exist:
        return all(os.path.getsize(os.path.join(folder, req)) > 0 for req in REQUIRED_FILES)
    else:
        return False


def find_calculation_directories(folder: str) -> Iterable[str]:
    return map(itemgetter(0), filter(star(is_vasp_calc_dir), os.walk(folder)))


def parse_structure(folder: str) -> Structure:
    return Poscar.from_file(os.path.join(folder, "CONTCAR")).structure


def parse_force_vector(x: str, y: str, z: str, fx: str, fy: str, fz: str) -> Vector:
    return float(fx), float(fy), float(fz)


def parse_forces(folder: str, natoms: int, block_size: int = 1024000 // 2) -> Optional[List[Vector]]:
    outcar_path = os.path.join(folder, "OUTCAR")
    with ropen(outcar_path, block_size=block_size) as handle:
        data = itertools.takewhile(lambda line: FORCE_MARK.match(line) is None, handle)
        stream = reversed(list(data))
    assert re.match(r"^\s*-+", next(stream))
    try:
        forces: List[Vector] = [parse_force_vector(**FORCE_REGEX.match(next(stream)).groupdict()) for _ in
                                range(natoms)]
    except Exception:
        warnings.warn(f"Failed to parse forces for \"{folder}\"", RuntimeWarning)
        return None
    assert re.match(r"^\s*-+", next(stream))
    return forces


def parse_oszicar_line(line: str) -> Optional[re.Match]:
    m = SCF_STEP_REGEX.match(line)
    if m is None:
        m = IONIC_STEP_REGEX.match(line)
    return m


def parse_scf_step(algo: str, step: str, e: str, de: str, deps: str, ncg: str, rms: str, rmsc: Optional[str]) -> Dict[
    str, Union[float, int, str]]:
    r = dict(algo=algo, step=int(step), e=float(e), de=float(de), deps=float(deps), ncg=int(ncg), rms=float(rms))
    if rmsc is not None:
        r["rmsc"] = float(rmsc)
    return r


def parse_ionic_step_summary(step: str, f: str, ezero: str, de: str, mag: Optional[str]) -> Dict[
    str, Union[float, str]]:
    r = dict(step=int(step), f=float(f), ezero=float(ezero), de=float(de))
    if mag is not None:
        r["mag"] = float(mag)
    return r


def parse_oszicar_block(block: Iterable[re.Match]) -> Dict[str, Any]:
    *scf, summary = block
    try:
        data = parse_ionic_step_summary(**summary.groupdict())
    except Exception:
        warnings.warn("Calculation did no converge", RuntimeWarning)
        data = dict()
    data["scf"] = [parse_scf_step(**scf_step.groupdict()) for scf_step in scf]
    return data


def parse_energies(folder: str) -> NoReturn:
    with open(os.path.join(folder, "OSZICAR")) as fh:
        lines: List[str] = list(fh)
    parsed_lines = map(parse_oszicar_line, lines)
    ionic_steps: List[List[re.Match]] = [list(ionic_step) for is_a_match, ionic_step in
                                         itertools.groupby(parsed_lines, key=isa(re.Match)) if is_a_match]
    return list(map(parse_oszicar_block, ionic_steps))


def parse(folder: str, progress: bool = False, forces: bool = True, energies: bool = True) -> Dict[str, Record]:
    folders = find_calculation_directories(folder)
    data = dict()

    for calc_folder in tqdm.tqdm(list(folders)) if progress else folders:
        record = dict()
        if forces:
            structure = parse_structure(calc_folder)
            forces_ = parse_forces(calc_folder, len(structure))
            if forces_ is not None:
                for axes, force in zip(("fx", "fy", "fz"), transpose(forces_)):
                    structure.add_site_property(axes, force)
            record["structure"] = structure
        if energies:
            energies_ = parse_energies(calc_folder)
            record["energies"] = energies_
        data[calc_folder] = record
    return data


def summary_force(record: Record) -> Dict[str, float]:
    structure = record.get("structure", None)
    force_record = dict()
    if structure is not None:
        forces = np.vstack([structure.site_properties[f"f{axis}"] for axis in "xyz"]).T
        fx, fy, fz = map(float, np.amax(forces, axis=0))
        maxf = float(np.amax(np.linalg.norm(forces, axis=1)))
        force_record.update(fx=fx, fy=fy, fz=fz, maxfnorm=maxf, log10maxf=math.log10(maxf))
    return force_record


def update_if_present(dst: Dict[str, float], key: str, src: Dict[str, Any], entry_name: Optional[str] = None,
                      operation: Callable[[Any], Any] = identity, default: Optional[Any] = None) -> NoReturn:
    dst[entry_name or key] = operation(src.get(key)) if key in src else default


def summary_energies(record: Record) -> Dict[str, float]:
    energies = record.get("energies", None)
    energy_record = dict()
    if energies is not None:

        last_step = energies[-1]
        energy_record["steps"] = len(energies)
        update_if_present(energy_record, "f", last_step)
        update_if_present(energy_record, "de", last_step, entry_name="de-ionic")
        update_if_present(energy_record, "de", last_step, entry_name="log10-de-ionic",
                          operation=lambda de: math.log10(abs(de)))
        update_if_present(energy_record, "mag", last_step)

        scf_steps = last_step.get("scf")
        if len(scf_steps):
            last_scf_step = scf_steps[-1]
            update_if_present(energy_record, "de", last_scf_step, entry_name="de-elec")
            update_if_present(energy_record, "de", last_scf_step, entry_name="log10-de-elec",
                              operation=lambda de: math.log10(abs(de)))

    return energy_record


def summary(folder: str, progress: bool = False, forces: bool = True, energies: bool = True) -> pd.DataFrame:
    summary_data = []
    parsed_data = parse(folder, progress=progress, forces=forces, energies=energies)
    for folder, parsed_data_record in parsed_data.items():
        summary_record = dict(folder=folder)
        if forces:
            summary_record.update(summary_force(parsed_data_record))
        if energies:
            summary_record.update(summary_energies(parsed_data_record))
        summary_data.append(summary_record)
    return pd.DataFrame(data=summary_data), parsed_data


def export_data(filename: str, data: Dict[str, Any], represenation: str = "pickle",
                compression: Optional[str] = None) -> NoReturn:
    open_function = COMPRESSIONS.get(compression)
    dumps = REPRESENTERS.get(represenation)

    # prepare data for export -> convert pymatgen.core.Structure to dict
    data = data.copy()
    for record in data.values():
        update_if_present(record, "structure", record, operation=lambda structure: structure.as_dict())

    with open_function(filename, "wb") as dump_handle:
        dump_handle.write(dumps(data))


def import_data(filename: str, represenation: str = "pickle", compression: Optional[str] = None) -> Dict[str, Any]:
    open_function = COMPRESSIONS.get(compression)
    loads = LOADERS.get(represenation)

    with open_function(filename, "rb") as dump_handle:
        data = loads(dump_handle.read())
    for record in data.values():
        update_if_present(record, "structure", record, operation=lambda structure: Structure.from_dict(structure))
    return data


@click.command()
@click.argument("folder", type=click.Path(file_okay=False), required=True)
@click.option("--progress/--no-progress", "-p/-np", default=False, help="display a progress bar")
@click.option("--forces/--no-forces", "-f/-nf", default=True,
              help="include or omit forces in the summary (default is True)")
@click.option("--energies/--no-energies", "-e/-ne", default=True,
              help="include or omit energies in the summary (default is True)")
@click.option("--export", "-e", "filename", type=click.Path(), help="filename to which to export the parsed data")
@click.option("--represenation", "-r", type=click.Choice(list(REPRESENTERS)), default="pickle",
              help="output format for the parsed data")
@click.option("--compression", "-c", type=click.Choice([method for method in COMPRESSIONS if method is not None]),
              help="compression algorithm for the output data")
def cli(folder: str, progress: bool = False, forces: bool = True, energies: bool = True, filename: Optional[str] = None,
        represenation: str = "pickle", compression: Optional[str] = None):
    summary_table, parsed_data = summary(folder, progress=progress, forces=forces, energies=energies)

    with pd.option_context('expand_frame_repr', True, 'display.max_rows', None):
        summary_table.folder = summary_table.folder.apply(lambda p: os.path.relpath(p, folder))
        print(summary_table)

    if filename is not None:
        filename = f"{filename}.{represenation}"
        if compression is not None:
            filename = f"{filename}.{compression}"
        export_data(filename, parsed_data, represenation=represenation, compression=compression)


if __name__ == "__main__":
    cli()
