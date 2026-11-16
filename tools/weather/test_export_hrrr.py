import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from pyproj import Geod, Proj

from export_hrrr import earth_wind, inventory, message


class WeatherExtractionChecks(unittest.TestCase):
    def test_map_grid_wind_matches_geographic_displacement(self):
        project = Proj(proj='lcc', lat_1=38.5, lat_2=38.5, lat_0=38.5,
                       lon_0=-97.5, R=6371229)
        geodesic = Geod(a=6371229, b=6371229)
        for longitude in [-110, -97.5, -94.46, -80]:
            latitude = 41.31
            x, y = project(longitude, latitude)
            angle = np.deg2rad(project.get_factors(longitude, latitude).meridian_convergence)
            for u, v in [(1, 0), (0, 1), (-3, 4)]:
                east, north = earth_wind(u, v, angle)
                end_lon, end_lat = project(x + u, y + v, inverse=True)
                azimuth, _, _ = geodesic.inv(longitude, latitude, end_lon, end_lat)
                expected = np.deg2rad(azimuth)
                speed = np.hypot(u, v)
                np.testing.assert_allclose([east / speed, north / speed],
                                           [np.sin(expected), np.cos(expected)], atol=1e-6)
                self.assertAlmostEqual(np.hypot(east, north), speed, places=12)

    def test_byte_range_has_no_neighboring_message(self):
        records = inventory('1:0:d=x:TMP:1 hybrid level:anl:\n2:123:d=x:SPFH:1 hybrid level:anl:\n')
        self.assertEqual(records[0]['end'], 122)
        with tempfile.TemporaryDirectory() as directory:
            with patch('export_hrrr.requests.get') as get:
                get.return_value.status_code = 206
                get.return_value.content = b'x' * 123
                result = message('https://example.invalid/model', records[0], Path(directory))
                self.assertEqual(len(result), 123)
                self.assertEqual(get.call_args.kwargs['headers']['Range'], 'bytes=0-122')
                message('https://example.invalid/model', records[0], Path(directory))
                self.assertEqual(get.call_count, 1)

    def test_full_file_response_cannot_masquerade_as_one_message(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('export_hrrr.requests.get') as get:
                get.return_value.status_code = 200
                get.return_value.content = b'GRIB'
                with self.assertRaisesRegex(ValueError, 'byte range'):
                    message('https://example.invalid/model', {'offset': 0, 'end': 3}, Path(directory))


if __name__ == '__main__':
    unittest.main()
