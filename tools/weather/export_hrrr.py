"""Extract an authentic regional HRRR state without downloading the whole model file."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import requests


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', default='20240521')
    parser.add_argument('--hour', type=int, default=18)
    parser.add_argument('--forecast', type=int, default=0)
    parser.add_argument('--latitude', type=float, default=41.31)
    parser.add_argument('--longitude', type=float, default=-94.46)
    parser.add_argument('--width-km', type=float, default=256)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--python-packages', type=Path)
    parser.add_argument('--levels', type=int, nargs='+', default=list(range(1, 51)))
    return parser.parse_args()


def inventory(text):
    records = []
    for line in text.splitlines():
        parts = line.split(':')
        records.append({'offset': int(parts[1]), 'variable': parts[3], 'level': parts[4]})
    for current, following in zip(records, records[1:]):
        current['end'] = following['offset'] - 1
    return records


def message(url, record, cache):
    path = cache / f"{record['offset']}.grib2"
    expected = record['end'] - record['offset'] + 1
    if not path.exists() or path.stat().st_size != expected:
        response = requests.get(url, headers={
            'Range': f"bytes={record['offset']}-{record['end']}"
        }, timeout=90)
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != expected:
            raise ValueError('The archive did not return the requested GRIB byte range.')
        temporary = path.with_suffix('.partial')
        temporary.write_bytes(response.content)
        temporary.replace(path)
    return path.read_bytes()


def earth_wind(u, v, convergence):
    cosine = np.cos(convergence)
    sine = np.sin(convergence)
    return u * cosine + v * sine, -u * sine + v * cosine


def main():
    args = arguments()
    if args.python_packages:
        sys.path.append(str(args.python_packages.resolve()))
    import eccodes as ec
    from pyproj import Proj

    name = f'hrrr.t{args.hour:02d}z.wrfnatf{args.forecast:02d}.grib2'
    url = f'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{args.date}/conus/{name}'
    cache = args.cache / args.date / name
    cache.mkdir(parents=True, exist_ok=True)
    index_path = cache / 'inventory.idx'
    if not index_path.exists():
        response = requests.get(url + '.idx', timeout=30)
        response.raise_for_status()
        index_path.write_text(response.text)
    records = inventory(index_path.read_text())
    variables = ['PRES', 'HGT', 'TMP', 'SPFH', 'UGRD', 'VGRD', 'VVEL',
                 'CLMR', 'RWMR', 'CIMIXR', 'SNMR', 'GRLE']
    levels = sorted(set(args.levels))
    selected = [record for record in records if record['variable'] in variables
                and record['level'] in {f'{level} hybrid level' for level in levels}]
    selected += [next(record for record in records
                      if record['variable'] == 'HGT' and record['level'] == 'surface')]
    if len(selected) != len(levels) * len(variables) + 1:
        raise ValueError('The archive inventory is missing a requested native-level field.')

    first = ec.codes_new_from_message(message(url, selected[0], cache))
    def get(key):
        return ec.codes_get(first, key)
    if get('gridType') != 'lambert' or get('jScansPositively') != 1:
        raise ValueError('Unsupported HRRR grid orientation.')
    if get('iScansNegatively') or get('jPointsAreConsecutive') or get('alternativeRowScanning'):
        raise ValueError('Unsupported GRIB storage order.')
    nx, ny = get('Nx'), get('Ny')
    spacing = [get('DxInMetres'), get('DyInMetres')]
    projection = dict(proj='lcc', lat_1=get('Latin1InDegrees'), lat_2=get('Latin2InDegrees'),
                      lat_0=get('LaDInDegrees'), lon_0=get('LoVInDegrees') - 360,
                      R=get('radius'))
    project = Proj(**projection)
    origin = project(get('longitudeOfFirstGridPointInDegrees') - 360,
                     get('latitudeOfFirstGridPointInDegrees'))
    center = project(args.longitude, args.latitude)
    radius = args.width_km * 500
    lower = [int(np.floor((center[i] - radius - origin[i]) / spacing[i])) for i in range(2)]
    upper = [int(np.ceil((center[i] + radius - origin[i]) / spacing[i])) + 1 for i in range(2)]
    if lower[0] < 0 or lower[1] < 0 or upper[0] > nx or upper[1] > ny:
        raise ValueError('Requested region extends outside the HRRR grid.')
    x = origin[0] + np.arange(lower[0], upper[0]) * spacing[0]
    y = origin[1] + np.arange(lower[1], upper[1]) * spacing[1]
    longitude, latitude = project(*np.meshgrid(x, y), inverse=True)
    convergence = np.deg2rad(project.get_factors(longitude, latitude).meridian_convergence)
    relative = get('uvRelativeToGrid')
    ec.codes_release(first)

    provenance = []
    fields = {}
    def extract(record, raw):
        handle = ec.codes_new_from_message(raw)
        try:
            values = ec.codes_get_values(handle).reshape(ny, nx)
            crop = values[lower[1]:upper[1], lower[0]:upper[0]].astype(np.float32)
            if not np.isfinite(crop).all() or ec.codes_get(handle, 'numberOfMissing'):
                raise ValueError(f"Missing values in {record['variable']} {record['level']}")
            units = ec.codes_get(handle, 'units')
            # Older NCEP files use this parameter with a master table predating its ecCodes entry.
            identity = [ec.codes_get(handle, key) for key in
                        ['discipline', 'parameterCategory', 'parameterNumber']]
            if record['variable'] == 'CIMIXR' and units == 'unknown' and identity == [0, 1, 82]:
                units = 'kg kg**-1'
            return record, crop, units, hashlib.sha256(raw).hexdigest()
        finally:
            ec.codes_release(handle)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(selected), 8):
            batch = selected[start:start + 8]
            downloads = pool.map(lambda record: message(url, record, cache), batch)
            # ecCodes decoding stays on one thread; its Windows codec DLLs are not reentrant.
            for record, raw in zip(batch, downloads):
                record, crop, units, checksum = extract(record, raw)
                fields[(record['variable'], record['level'])] = crop
                provenance.append({**record, 'units': units, 'sha256': checksum})
            print(f'{min(start + 8, len(selected))}/{len(selected)} fields extracted', flush=True)

    def stack(variable):
        return np.stack([fields[(variable, f'{level} hybrid level')] for level in levels])

    expected_units = {'PRES': 'Pa', 'HGT': 'gpm', 'TMP': 'K', 'SPFH': 'kg kg**-1',
                      'UGRD': 'm s**-1', 'VGRD': 'm s**-1', 'VVEL': 'Pa s**-1'}
    for record in provenance:
        expected = expected_units.get(record['variable'])
        if record['variable'] == 'HGT' and record['level'] == 'surface':
            expected = 'm'
        if expected and record['units'] != expected:
            raise ValueError(f"Unexpected units for {record['variable']}: {record['units']}")
        if record['variable'] in ['CLMR', 'RWMR', 'CIMIXR', 'SNMR', 'GRLE']:
            if record['units'] != 'kg kg**-1':
                raise ValueError(f"Unexpected hydrometeor units: {record['units']}")

    pressure, temperature, humidity = stack('PRES'), stack('TMP'), stack('SPFH')
    density = pressure / (287.05 * temperature * (1 + (461.5 / 287.05 - 1) * humidity))
    u, v = stack('UGRD'), stack('VGRD')
    if relative:
        u, v = earth_wind(u, v, convergence)
    geopotential_height = stack('HGT')
    geometric_height = projection['R'] * geopotential_height / (projection['R'] - geopotential_height)
    if len(levels) > 1 and (not np.all(np.diff(geometric_height, axis=0) > 0)
                            or not np.all(np.diff(pressure, axis=0) < 0)):
        raise ValueError('Native levels must rise monotonically while pressure decreases.')
    output = {
        'x': x, 'y': y, 'latitude': latitude, 'longitude': longitude,
        'terrain': fields[('HGT', 'surface')], 'height': geometric_height,
        'pressure': pressure, 'temperature': temperature, 'specific_humidity': humidity,
        'east_wind': u, 'north_wind': v, 'vertical_wind': -stack('VVEL') / (density * 9.80665),
        'cloud_water': stack('CLMR'), 'rain_water': stack('RWMR'),
        'cloud_ice': stack('CIMIXR'), 'snow': stack('SNMR'), 'graupel': stack('GRLE')
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    metadata = {
        'source': url, 'initializationUtc': datetime.strptime(
            f'{args.date}{args.hour:02d}', '%Y%m%d%H').replace(tzinfo=timezone.utc).isoformat(),
        'forecastHours': args.forecast, 'nativeSpacingMeters': spacing,
        'projection': projection, 'nativeLevels': levels, 'center': [args.latitude, args.longitude],
        'arrayOrder': 'native level (bottom to top), south to north, west to east',
        'windConversion': 'grid-relative to true east/north using Lambert meridian convergence',
        'verticalWindConversion': 'w = -omega / (moist-air density * standard gravity)',
        'heightConversion': 'Native geopotential height H becomes geometric height R*H/(R-H). Surface orography is already in meters.',
        'syntheticPerturbations': False,
        'parameterDefinitions': 'https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-1.shtml',
        'limitations': ['Model analysis/forecast, not direct observations.',
                        'Interpolation cannot add spatial information below the 3 km source grid.',
                        'Retained ice fields require ice physics to evolve faithfully.'],
        'statistics': {key: {'min': float(value.min()), 'max': float(value.max())}
                       for key, value in output.items()},
        'messages': provenance
    }
    args.output.with_suffix('.source.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata['statistics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
