"""Pack extracted HRRR snapshots for the browser's spatial weather experiment."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ground-msl', type=float, default=200)
    parser.add_argument('--latitude', type=float)
    parser.add_argument('--longitude', type=float)
    args = parser.parse_args()
    if not 2 <= len(args.snapshots) <= 4:
        raise ValueError('Supply two to four hourly snapshots for the browser input budget.')
    sources = [json.loads(path.with_suffix('.source.json').read_text()) for path in args.snapshots]
    data = [np.load(path) for path in args.snapshots]
    first = data[0]
    projection = sources[0]['projection']
    levels, north, east = first['height'].shape
    if levels != 50:
        raise ValueError('The browser experiment needs all 50 native levels.')
    for current, source in zip(data, sources):
        if source['projection'] != projection or current['height'].shape != first['height'].shape:
            raise ValueError('Snapshots must use the same native grid.')
        for axis in ['x', 'y']:
            if not np.array_equal(current[axis], first[axis]):
                raise ValueError('Snapshot coordinate axes differ.')
    forecast_hours = [source['forecastHours'] for source in sources]
    if any(source['initializationUtc'] != sources[0]['initializationUtc'] for source in sources):
        raise ValueError('Use consecutive forecasts from one initialization, not unrelated analyses.')
    if forecast_hours != list(range(forecast_hours[0], forecast_hours[0] + len(sources))):
        raise ValueError('Snapshots must be consecutive hourly forecast steps.')

    latitude, longitude = sources[0]['center']
    latitude = latitude if args.latitude is None else args.latitude
    longitude = longitude if args.longitude is None else args.longitude
    n = np.sin(np.deg2rad(projection['lat_1']))
    factor = np.cos(np.deg2rad(projection['lat_1'])) * (
        np.tan(np.pi / 4 + np.deg2rad(projection['lat_1']) / 2) ** n) / n
    rho0 = projection['R'] * factor / (
        np.tan(np.pi / 4 + np.deg2rad(projection['lat_0']) / 2) ** n)
    header = np.array([
        east, north, levels, len(data), first['x'][0], first['y'][0],
        first['x'][1] - first['x'][0], first['y'][1] - first['y'][0],
        n, factor, rho0, projection['R'], np.deg2rad(latitude), np.deg2rad(longitude),
        args.ground_msl, np.deg2rad(projection['lon_0']),
        0, 3600, 2, 0, 0, 0, 0, 0
    ], dtype='<f4')
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'state.bin'
    with target.open('wb') as stream:
        header.tofile(stream)
        for current in data:
            packed = np.zeros((levels, north, east, 16), dtype='<f4')
            packed[..., 0] = current['height']
            packed[..., 1] = current['temperature'] * (100000 / current['pressure']) ** (287.05 / 1004)
            packed[..., 2] = 1000 * current['specific_humidity'] / (1 - current['specific_humidity'])
            packed[..., 3] = current['east_wind']
            packed[..., 4] = -current['north_wind']
            packed[..., 5] = current['vertical_wind']
            packed[..., 6] = 1000 * current['cloud_water']
            packed[..., 7] = 1000 * current['rain_water']
            packed[..., 8] = current['pressure']
            packed[..., 9] = current['temperature']
            q = current['specific_humidity']
            packed[..., 10] = current['pressure'] / (287.05 * current['temperature'] * (1 + 0.608 * q))
            packed[..., 11] = 1000 * (current['cloud_ice'] + current['snow'] + current['graupel'])
            packed[..., 12] = current['terrain']
            packed[..., 13] = current['turbulent_energy']
            if not np.isfinite(packed).all():
                raise ValueError('Non-finite weather input.')
            packed.tofile(stream)
        for current, source in zip(data, sources):
            for variable in ['SHTFL', 'LHTFL']:
                record = next(record for record in source['messages']
                              if record['variable'] == variable and record['level'] == 'surface')
                if record['stepType'] != 'instant':
                    raise ValueError('This browser format expects instantaneous surface fluxes.')
            surface = np.zeros((north, east, 8), dtype='<f4')
            surface[..., 0] = current['roughness']
            surface[..., 1] = current['sensible_heat_flux']
            surface[..., 2] = current['latent_heat_flux']
            surface[..., 3] = current['friction_velocity']
            surface.tofile(stream)
    manifest = {
        'version': 2, 'state': 'state.bin', 'bytes': target.stat().st_size,
        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'center': {'latitude': latitude, 'longitude': longitude},
        'groundMsl': args.ground_msl, 'shape': [east, levels, north],
        'frames': len(data), 'intervalSeconds': 3600, 'forecastHours': forecast_hours,
        'sources': sources,
        'limitations': ['Warm-rain physics cannot evolve the retained ice species.',
                        'The browser uses a flat map and voxel terrain; it is not the HRRR dynamical core.',
                        'Only lateral boundaries receive later forecasts. The interior evolves freely.']
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Packed {len(data)} frames, {target.stat().st_size / 1048576:.1f} MiB: {target}')


if __name__ == '__main__':
    main()
