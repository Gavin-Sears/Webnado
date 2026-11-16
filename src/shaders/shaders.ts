// CHECKITOUT: loads all the shaders and prepends common code

import commonRaw from './common.wgsl?raw';

import bricksComputeRaw from './bricks.cs.wgsl?raw';
import compositeFragRaw from './composite.fs.wgsl?raw';
import fullscreenVertRaw from './fullscreen.vs.wgsl?raw';
import raymarchFragRaw from './raymarch.fs.wgsl?raw';
import testVolumeComputeRaw from './test_volume.cs.wgsl?raw';

// CONSTANTS (for use in shaders)
// =================================

// CHECKITOUT: add constants here to use them in shader code

// Dimensions must be multiples of 8 (M3's brick size), and cells should stay cubes
export const resolutions: Record<string, [number, number, number]> = {
    low: [64, 32, 64],
    medium: [128, 64, 128],
    high: [256, 128, 256]
};

const requestedResolution = new URLSearchParams(window.location.search).get('res') ?? 'medium';
export const resolutionName = Object.hasOwn(resolutions, requestedResolution) ? requestedResolution : 'medium';
const [gridX, gridY, gridZ] = resolutions[resolutionName];
export const domains = {
    storm: [12800, 6400, 12800],
    local: [3200, 1600, 3200],
    supercell: [64000, 18000, 64000]
} as const;
const requestedDomain = new URLSearchParams(window.location.search).get('domain')
    ?? (new URLSearchParams(window.location.search).get('case') === 'elreno' ? 'supercell' : 'storm');
export const domainName = Object.hasOwn(domains, requestedDomain) ? requestedDomain as keyof typeof domains : 'storm';
const [domainX, domainY, domainZ] = domains[domainName];

// Declared this way so minification keeps values that only shaders use
export const constants = {
    gridX,
    gridY,
    gridZ,

    // box size in meters, centered on x = z = 0 with the ground at y = 0
    domainX,
    domainY,
    domainZ,

    atmosphereSlices: domainName === 'supercell' ? 256 : 64,
    freeSlipGround: Number(domainName === 'supercell'),

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
export const raymarchFragSrc: string = processShaderRaw(raymarchFragRaw);
export const compositeFragSrc: string = processShaderRaw(compositeFragRaw);
export const bricksComputeSrc: string = processShaderRaw(bricksComputeRaw);
export const testVolumeComputeSrc: string = processShaderRaw(testVolumeComputeRaw);
