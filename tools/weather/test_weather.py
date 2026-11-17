"""Fast weather-tool checks; these do not download model files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from export_hrrr import inventory
from prepare import identifier, prepare, validate

class WeatherChecks(unittest.TestCase):
    def settings(self):
        return dict(date='2024-05-21',hour=18,forecast=2,latitude=40.93,longitude=-95.01)

    def test_inventory_ranges(self):
        result=inventory('1:0:d=2024052118:TMP:1 hybrid level:\n2:128:d=2024052118:PRES:1 hybrid level:')
        self.assertEqual(result[0]['end'],127)
        self.assertEqual(result[1]['offset'],128)

    def test_request_validation(self):
        for changes in [dict(latitude=90),dict(longitude=0),dict(hour=24),dict(forecast=17),dict(date='2024-02-30')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate(self.settings()|changes)
        self.assertEqual(validate(self.settings())['forecast'],2)
        self.assertNotEqual(identifier(validate(self.settings())),identifier(validate(self.settings()|dict(hour=17))))

    def test_cached_preparation_never_downloads(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/identifier(validate(self.settings()))
            target.mkdir();(target/'manifest.json').write_text('{}')
            with patch('prepare.subprocess.run') as run:
                self.assertEqual(prepare(self.settings(),folder),target)
                run.assert_not_called()

    def test_cached_service_and_path_rejection(self):
        import service
        with tempfile.TemporaryDirectory() as folder:
            key=identifier(validate(self.settings()))
            target=Path(folder)/key;target.mkdir()
            (target/'manifest.json').write_text('{"state":"state.bin"}')
            (target/'state.bin').write_bytes(b'weather fixture')
            captured=[]
            real=service.ThreadingHTTPServer
            def server(address,handler):
                result=real(('127.0.0.1',0),handler);captured.append(result);return result
            with patch.object(service,'ThreadingHTTPServer',server):
                thread=threading.Thread(target=service.serve,args=(folder,'127.0.0.1',0,'https://example.test'),daemon=True)
                thread.start()
                import time
                for _ in range(100):
                    if captured:break
                    time.sleep(.01)
                host=captured[0];base=f'http://127.0.0.1:{host.server_port}'
                try:
                    request=Request(base+'/jobs',data=json.dumps(self.settings()).encode(),headers={'Content-Type':'application/json'})
                    with urlopen(request) as response:
                        job=json.load(response)
                        self.assertEqual(response.headers['Access-Control-Allow-Origin'],'https://example.test')
                    self.assertEqual(job['status'],'ready')
                    with urlopen(base+'/'+job['manifest']) as response:self.assertIn(b'state.bin',response.read())
                    with self.assertRaises(HTTPError):urlopen(base+'/data/../../requirements.txt')
                finally:host.shutdown();host.server_close();thread.join(3)

if __name__=='__main__':unittest.main()
