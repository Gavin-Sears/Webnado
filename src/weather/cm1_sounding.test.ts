import { cm1Atmosphere, parseCm1Sounding } from './cm1_sounding';
import elReno from './cases/elreno-2011.txt?raw';

const output = document.getElementById('results')!;

function expect(condition: boolean, message: string) {
    if (!condition) {
        throw new Error(message);
    }
    output.textContent += `\nPASS ${message}`;
}

try {
    const parsed = parseCm1Sounding(elReno);
    expect(parsed.points.length === 153 && parsed.points.at(-1)!.height === 18000,
        'The published sounding retains its full depth');
    expect(parsed.points[0].windX === -2.23 && parsed.points[0].windZ === -9.06,
        'East/north winds map to the project coordinate system');

    const dry = parseCm1Sounding('1000 300 0\n100 300 0 4 -6\n1000 300 0 4 -6');
    const exner = 1 - 9.81 * 1000 / (1004 * 300);
    const expected = 100000 * Math.pow(exner, 1004 / 287.05);
    expect(Math.abs(dry.points[1].pressure - expected) < 0.01,
        'Hydrostatic pressure matches the analytical dry constant-theta column');

    const moist = parseCm1Sounding('1000 300 20\n100 300 20 0 0\n1000 300 20 0 0');
    expect(moist.points[1].pressure > dry.points[1].pressure,
        'Moisture changes the hydrostatic column through virtual temperature');

    for (let i = 1; i < parsed.points.length; i++) {
        const point = parsed.points[i];
        expect(Number.isFinite(point.density) && point.density > 0
            && point.pressure < parsed.points[i - 1].pressure, `Hydrostatic level ${i} is physical`);
    }
    const column = cm1Atmosphere(elReno, 128, 18000);
    expect(column.length === 1024 && column.every(Number.isFinite),
        'The complete experiment profile fits the shared atmosphere layout');
    let rejected = false;
    try {
        cm1Atmosphere(elReno, 128, 20000);
    } catch {
        rejected = true;
    }
    expect(rejected, 'A domain above the available sounding is rejected');
    output.textContent += '\nAll checks passed.';
} catch (error) {
    output.textContent += `\nFAIL ${error}`;
}
