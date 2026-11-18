"""Prepare immutable, browser-ready HRRR cases for a worker or static deployment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent

def validate(value):
    date = datetime.strptime(value['date'], '%Y-%m-%d').date()
    if date < datetime(2020, 12, 2).date() or date > datetime.now(timezone.utc).date():
        raise ValueError('Choose an HRRR v4 date from December 2020 through today.')
    result = dict(date=date.isoformat(), hour=int(value['hour']), forecast=int(value.get('forecast', 1)),
                  latitude=float(value['latitude']), longitude=float(value['longitude']))
    if not 0 <= result['hour'] <= 23 or not 1 <= result['forecast'] <= 16:
        raise ValueError('Cycle hour must be 0-23 UTC and first forecast hour 1-16.')
    if not 25 <= result['latitude'] <= 49 or not -124 <= result['longitude'] <= -67:
        raise ValueError('This importer supports the contiguous United States only.')
    if 'groundMsl' in value:
        result['groundMsl'] = float(value['groundMsl'])
        if not -500 <= result['groundMsl'] <= 9000:
            raise ValueError('Invalid comparison floor height')
    return result

def identifier(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:24]

def prepare(settings, output, progress=print):
    import numpy as np
    settings = validate(settings)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / identifier(settings)
    if (target / 'manifest.json').exists():
        return target
    if shutil.disk_usage(output).free < 8 * 1024**3:
        raise ValueError('Weather preparation needs at least 8 GiB free temporary storage.')
    with tempfile.TemporaryDirectory(prefix='hrrr-', dir=output) as work:
        work = Path(work)
        snapshots = []
        for lead in range(settings['forecast'], settings['forecast'] + 3):
            progress(f'Extracting forecast hour {lead}; this can take several minutes')
            snapshot = work / f'forecast-{lead:02}.npz'
            command = [sys.executable, str(ROOT / 'export_hrrr.py'), '--date', settings['date'].replace('-', ''),
                       '--hour', str(settings['hour']), '--forecast', str(lead),
                       '--latitude', str(settings['latitude']), '--longitude', str(settings['longitude']),
                       '--width-km', '256', '--output', str(snapshot), '--cache', str(work / 'raw')]
            subprocess.run(command, check=True, timeout=1800)
            snapshots.append(snapshot)
        with np.load(snapshots[0]) as data:
            distance = (data['latitude'] - settings['latitude'])**2 + (
                (data['longitude'] - settings['longitude']) * np.cos(np.deg2rad(settings['latitude'])))**2
            index = np.unravel_index(distance.argmin(), distance.shape)
            ground = settings.get('groundMsl', float(data['terrain'][index]))
        progress('Packing browser data')
        packed = work / 'packed'
        subprocess.run([sys.executable, str(ROOT / 'pack_hrrr.py'), *map(str, snapshots),
                        '--output', str(packed), '--ground-msl', str(ground),
                        '--latitude', str(settings['latitude']), '--longitude', str(settings['longitude'])],
                       check=True, timeout=300)
        (packed / 'request.json').write_text(json.dumps(settings, indent=2))
        packed.rename(target)
    return target

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT / 'cases.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    catalog = []
    for case in json.loads(args.cases.read_text()):
        target = prepare(case, args.output)
        catalog.append({'name': case['name'], 'manifest': f'{target.name}/manifest.json'})
    temporary = args.output / 'catalog.tmp'
    temporary.write_text(json.dumps(catalog, indent=2))
    temporary.replace(args.output / 'catalog.json')

if __name__ == '__main__':
    main()
