// CHECKITOUT: loads all the shaders and prepends common code

import commonRaw from './common.wgsl?raw';

import bricksComputeRaw from './bricks.cs.wgsl?raw';
import compositeFragRaw from './composite.fs.wgsl?raw';
import fullscreenVertRaw from './fullscreen.vs.wgsl?raw';
import raymarchFragRaw from './raymarch.fs.wgsl?raw';
import temporalFragRaw from './temporal.fs.wgsl?raw';
import testVolumeComputeRaw from './test_volume.cs.wgsl?raw';

// CONSTANTS (for use in shaders)
// =================================

// CHECKITOUT: add constants here to use them in shader code

// Dimensions must be multiples of 8 (M3's brick size), and cells should stay cubes
export const resolutions: Record<string, [number, number, number]> = {
    low: [64, 32, 64],
    medium: [128, 64, 128],
    balanced: [192, 96, 192],
    high: [256, 128, 256],
    fine: [384, 128, 384],
    vertical: [256, 256, 256],
    dense: [320, 192, 320]
};

const requestedResolution = new URLSearchParams(window.location.search).get('res') ?? 'medium';
export const resolutionName = Object.hasOwn(resolutions, requestedResolution) ? requestedResolution : 'medium';
const [gridX, gridY, gridZ] = resolutions[resolutionName];
export const domains = {
    storm: [12800, 6400, 12800],
    local: [3200, 1600, 3200],
    supercell: [64000, 18000, 64000],
    'storm-focus': [32000, 18000, 32000]
} as const;
const requestedDomain = new URLSearchParams(window.location.search).get('domain')
    ?? (['elreno', 'hrrr'].includes(new URLSearchParams(window.location.search).get('case') ?? '') ? 'supercell' : 'storm');
export const domainName = Object.hasOwn(domains, requestedDomain) ? requestedDomain as keyof typeof domains : 'storm';
const [domainX, domainY, domainZ] = domains[domainName];
const weatherQuery = new URLSearchParams(window.location.search);
const spatialCase = weatherQuery.get('case') === 'hrrr';
export const tkeModelName = weatherQuery.get('tke') === 'deardorff' ? 'deardorff' : 'legacy';
const atmosphericTke = tkeModelName === 'deardorff';

// Declared this way so minification keeps values that only shaders use
export const constants = {
    gridX,
    gridY,
    gridZ,

    // box size in meters, centered on x = z = 0 with the ground at y = 0
    domainX,
    domainY,
    domainZ,

    atmosphereSlices: domainName === 'supercell' || domainName === 'storm-focus' ? 256 : 64,
    freeSlipGround: Number(domainName === 'supercell' || domainName === 'storm-focus'),
    spatialWeather: Number(spatialCase),
    preciseScalars: Number(weatherQuery.get('state') === '32'),
    scalarFormat: weatherQuery.get('state') === '32' ? 'rgba32float' as const : 'rgba16float' as const,
    // Deardorff coefficients: https://doi.org/10.1175/MWR-2847.1
    tkeViscosity: atmosphericTke ? 0.1 : 0.2,
    tkeDissipation: atmosphericTke ? 0.7 : 0.2,
    tkeGeometricLength: Number(atmosphericTke),
    tkeScalarMixing: atmosphericTke ? 1 : 0.1,
    anelasticAir: Number(['elreno', 'hrrr'].includes(new URLSearchParams(window.location.search).get('case') ?? '')
        && (weatherQuery.get('projection') ?? (spatialCase ? 'anelastic' : 'volume')) === 'anelastic'),
    velocityFormat: (weatherQuery.get('precision') ?? (spatialCase ? '32' : '16')) === '32'
        ? 'rgba32float' as const : 'rgba16float' as const,
    dustSeedRadius: Math.min(1200, 0.15 * domainX),

    brickSize: 8    // cells per side of a brick, for skipping empty space
};

// =================================

function evalShaderRaw(raw: string): string {
    return raw.replace(
        /\$\{(\w+)\}/g,
        (_, name: string) => {
            const value = constants[name as keyof typeof constants];
            if (value === undefined) {
                throw new Error(`Missing shader constant: ${name}`);
            }

            return String(value);
        }
    );
}

const commonSrc: string = evalShaderRaw(commonRaw);

export function processShaderRaw(raw: string) {
    return commonSrc + evalShaderRaw(raw);
}

export const fullscreenVertSrc: string = processShaderRaw(fullscreenVertRaw);
const filteredThermo = 'return textureSampleLevel(thermo, linearSampler, uvw, 0.0);';

// In raymarch.fs.wgsl, always read thermo with sampleThermo(uvw) 
// and don't textureSample directly. Devices without float32-filterable 
// (32 float texture hardware sampling) will use sampleVolume() in the function body 
// instead, so 32-bit thermo filtering still works.
export function raymarchFragSrc(float32Filterable: boolean): string {
    if (constants.scalarFormat !== 'rgba32float' || float32Filterable) {
        return processShaderRaw(raymarchFragRaw);
    }
    const source = raymarchFragRaw.replace(filteredThermo, 'return sampleVolume(thermo, uvw);');
    if (source === raymarchFragRaw || /textureSample\w*\(\s*thermo/.test(source)) {
        throw new Error('raymarch.fs.wgsl must read thermo only through sampleThermo(), whose body must be: ' + filteredThermo);
    }
    return processShaderRaw(source);
}

export const compositeFragSrc: string = processShaderRaw(compositeFragRaw);
export const temporalFragSrc: string = processShaderRaw(temporalFragRaw);
export const bricksComputeSrc: string = processShaderRaw(bricksComputeRaw);
export const testVolumeComputeSrc: string = processShaderRaw(testVolumeComputeRaw);
