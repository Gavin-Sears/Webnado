import { atmosphereColumn, type SoundingPoint } from './sounding';

const gravity = 9.81;
const gasConstant = 287.05;
const specificHeat = 1004;
const kappa = gasConstant / specificHeat;

export interface Cm1Sounding {
    surfacePressure: number;
    points: SoundingPoint[];
}

export function parseCm1Sounding(text: string): Cm1Sounding {
    const rows = text.trim().split(/\r?\n/).map(line =>
        line.trim().split(/[\s,]+/).map(Number)
    );
    const header = rows.shift();
    if (!header || header.length !== 3 || rows.length < 2) {
        throw new Error('A CM1 sounding needs a surface header and at least two levels.');
    }
    const [pressureHpa, surfaceTheta, surfaceVapor] = header;
    if (!header.every(Number.isFinite) || pressureHpa <= 0 || surfaceTheta <= 0 || surfaceVapor < 0) {
        throw new Error('The sounding surface values are invalid.');
    }

    let height = 0;
    let theta = surfaceTheta;
    let vapor = surfaceVapor;
    let exner = Math.pow(pressureHpa * 100 / 100000, kappa);
    const points: SoundingPoint[] = [];
    const virtualTheta = (t: number, q: number) => t * (1 + q / 1000 / 0.622) / (1 + q / 1000);

    for (const row of rows) {
        const [nextHeight, nextTheta, nextVapor, east, north] = row;
        if (row.length !== 5 || !row.every(Number.isFinite) || nextHeight <= height
            || nextTheta <= 0 || nextVapor < 0) {
            throw new Error('Sounding levels must be finite and increase above the ground.');
        }
        const steps = Math.max(1, Math.ceil((nextHeight - height) / 25));
        const dz = (nextHeight - height) / steps;
        for (let step = 0; step < steps; step++) {
            const fraction = (step + 0.5) / steps;
            const t = theta + fraction * (nextTheta - theta);
            const q = vapor + fraction * (nextVapor - vapor);
            exner -= gravity * dz / (specificHeat * virtualTheta(t, q));
        }
        if (exner <= 0) {
            throw new Error('The sounding cannot support hydrostatic pressure at this height.');
        }
        const pressure = 100000 * Math.pow(exner, 1 / kappa);
        const temperature = nextTheta * exner;
        const virtualTemperature = temperature * virtualTheta(1, nextVapor);
        points.push({
            height: nextHeight, theta: nextTheta, vapor: nextVapor,
            windX: east, windZ: -north, pressure, temperature,
            density: pressure / (gasConstant * virtualTemperature)
        });
        height = nextHeight;
        theta = nextTheta;
        vapor = nextVapor;
    }
    return { surfacePressure: pressureHpa * 100, points };
}

export function cm1Atmosphere(text: string, slices: number, height: number): Float32Array {
    const sounding = parseCm1Sounding(text);
    if (height > sounding.points.at(-1)!.height) {
        throw new Error('The sounding must reach the top of the simulation.');
    }
    return atmosphereColumn(sounding.points, slices, height);
}
