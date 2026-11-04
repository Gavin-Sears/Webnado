// CHECKITOUT: loads all the shaders and prepends common code

import commonRaw from './common.wgsl?raw';

// these files are ones I am not committing yet because they need more work/time
// import fullscreenVertRaw from './fullscreen.vs.wgsl?raw';
// import raymarchFragRaw from './raymarch.fs.wgsl?raw';
// import testVolumeComputeRaw from './test_volume.cs.wgsl?raw';

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

// Declared this way so minification keeps values that only shaders use
export const constants = {
    gridX,
    gridY,
    gridZ,

    // box size in meters, centered on x = z = 0 with the ground at y = 0
    domainX: 12800,
    domainY: 6400,
    domainZ: 12800,

    atmosphereSlices: 64
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

// export const fullscreenVertSrc: string = processShaderRaw(fullscreenVertRaw);
// export const raymarchFragSrc: string = processShaderRaw(raymarchFragRaw);
// export const testVolumeComputeSrc: string = processShaderRaw(testVolumeComputeRaw);
