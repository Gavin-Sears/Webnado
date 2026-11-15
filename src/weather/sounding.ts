// One Open-Meteo hour, turned into the atmosphere buffer the solver already samples.
// Open-Meteo sends °C, %, hPa, and a wind direction (the direction the wind comes from).
// Each slice matches AtmosphereSample:
//   theta (K), qv (g/kg), wind x (m/s), wind z (m/s), pressure (Pa), temperature (K), density (kg/m^3)
// Grid theta' and qv' stay 0, which means "this sounding", until a parcel leaves it.

const PRESSURE_LEVELS_HPA = [1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400] as const;

export const VALUES_PER_SLICE = 8;

const REFERENCE_PRESSURE = 100000;
const POISSON_KAPPA = 0.2857;
const DRY_AIR = 287.05;

// One measured height, already converted to the units stored in the buffer.
export interface SoundingPoint {
    height: number;      // m above the ground
    theta: number;       // K, potential temperature
    vapor: number;       // g/kg
    windX: number;       // m/s toward the east
    windZ: number;       // m/s toward the south
    pressure: number;    // Pa
    temperature: number; // K
    density: number;     // kg/m^3
}

interface OpenMeteoHourly {
    time?: string[];
    [name: string]: Array<number | string | null> | undefined;
}

export interface OpenMeteoForecast {
    elevation?: number;
    hourly?: OpenMeteoHourly;
}

// Same Bolton expression as the microphysics shader, so the sounding and the saturation check agree.
function saturationVaporPressure(celsius: number): number {
    return 611.2 * Math.exp(17.67 * celsius / (celsius + 243.5));
}

// Grams of water vapor per kilogram of dry air.
function vaporContent(celsius: number, pressure: number, relativeHumidity: number): number {
    const es = saturationVaporPressure(celsius);
    const vaporPressure = (relativeHumidity / 100) * es;
    return 0.622 * vaporPressure / Math.max(pressure - vaporPressure, 1) * 1000;
}

// Potential temperature (K): the temperature the air would have if brought to 1000 hPa.
function potentialTemperature(kelvin: number, pressure: number): number {
    return kelvin * Math.pow(REFERENCE_PRESSURE / pressure, POISSON_KAPPA);
}

// Direction is where the wind comes from. +X is east and +Z is south,
// so a south wind has a negative Z component.
function windComponents(speed: number, directionDegrees: number): { windX: number, windZ: number } {
    const radians = directionDegrees * Math.PI / 180;
    const eastward = -speed * Math.sin(radians);
    const northward = -speed * Math.cos(radians);
    return { windX: eastward, windZ: -northward };
}

// Names for the hourly= list. Surface observations, then each pressure level.
export function hourlyNames(): string[] {
    const names = [
        'temperature_2m',
        'relative_humidity_2m',
        'surface_pressure',
        'wind_speed_10m',
        'wind_direction_10m'
    ];

    for (const level of PRESSURE_LEVELS_HPA) {
        names.push(
            `temperature_${level}hPa`,
            `relative_humidity_${level}hPa`,
            `wind_speed_${level}hPa`,
            `wind_direction_${level}hPa`,
            `geopotential_height_${level}hPa`
        );
    }

    return names;
}

function finite(hourly: OpenMeteoHourly, name: string, index: number): number | undefined {
    const column = hourly[name];
    const value = column?.[index];
    return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function pointAt(
    height: number,
    celsius: number,
    pressureHpa: number,
    relativeHumidity: number,
    speed: number,
    direction: number
): SoundingPoint {
    const pressure = pressureHpa * 100;
    const temperature = celsius + 273.15;
    return {
        height,
        theta: potentialTemperature(temperature, pressure),
        vapor: vaporContent(celsius, pressure, relativeHumidity),
        ...windComponents(speed, direction),
        pressure,
        temperature,
        density: pressure / (DRY_AIR * temperature)
    };
}

function hourIndex(times: string[], date: string, hour: number): number {
    const stamp = `${date}T${String(hour).padStart(2, '0')}:00`;
    return times.findIndex(time => time.startsWith(stamp));
}

// Levels that exist at this UTC hour. Surface is the 2 m temperature with the 10 m wind.
// A pressure level's height is geopotential height minus the station elevation, so y = 0 is the ground.
// Levels below the ground, and hours the forecast does not contain, are left out.
export function pointsFromHour(forecast: OpenMeteoForecast, date: string, hour: number): SoundingPoint[] {
    const hourly = forecast.hourly;
    const times = hourly?.time?.filter((time): time is string => typeof time === 'string') ?? [];
    const index = hourIndex(times, date, hour);
    if (!hourly || index < 0) {
        return [];
    }

    const elevation = typeof forecast.elevation === 'number' ? forecast.elevation : 0;
    const points: SoundingPoint[] = [];
    const surface = [
        finite(hourly, 'temperature_2m', index),
        finite(hourly, 'relative_humidity_2m', index),
        finite(hourly, 'surface_pressure', index),
        finite(hourly, 'wind_speed_10m', index),
        finite(hourly, 'wind_direction_10m', index)
    ];

    if (surface.every(value => value !== undefined)) {
        const [celsius, humidity, pressure, speed, direction] = surface as number[];
        points.push(pointAt(2, celsius, pressure, humidity, speed, direction));
    }

    for (const level of PRESSURE_LEVELS_HPA) {
        const values = [
            finite(hourly, `temperature_${level}hPa`, index),
            finite(hourly, `relative_humidity_${level}hPa`, index),
            finite(hourly, `geopotential_height_${level}hPa`, index),
            finite(hourly, `wind_speed_${level}hPa`, index),
            finite(hourly, `wind_direction_${level}hPa`, index)
        ];
        if (values.some(value => value === undefined)) {
            continue;
        }

        const [celsius, humidity, geometricHeight, speed, direction] = values as number[];
        const height = geometricHeight - elevation;
        if (height < 0) {
            continue;
        }

        points.push(pointAt(height, celsius, level, humidity, speed, direction));
    }

    points.sort((a, b) => a.height - b.height);
    return points;
}

function mix(lower: SoundingPoint, upper: SoundingPoint, weight: number): SoundingPoint {
    const blend = (a: number, b: number) => a + (b - a) * weight;
    return {
        height: blend(lower.height, upper.height),
        theta: blend(lower.theta, upper.theta),
        vapor: blend(lower.vapor, upper.vapor),
        windX: blend(lower.windX, upper.windX),
        windZ: blend(lower.windZ, upper.windZ),
        pressure: blend(lower.pressure, upper.pressure),
        temperature: blend(lower.temperature, upper.temperature),
        density: blend(lower.density, upper.density)
    };
}

// Outside the sounding, hold the nearest point. The top of the box can sit above 400 hPa.
function sampleColumn(points: SoundingPoint[], height: number): SoundingPoint {
    const first = points[0];
    const last = points[points.length - 1];
    if (height <= first.height) {
        return first;
    }
    if (height >= last.height) {
        return last;
    }

    let upper = 1;
    while (upper < points.length - 1 && points[upper].height < height) {
        upper++;
    }

    const below = points[upper - 1];
    const above = points[upper];
    const span = above.height - below.height;
    const weight = span > 0 ? (height - below.height) / span : 0;
    return mix(below, above, weight);
}

// A made-up column with the same temperature profile as the placeholder.
// vaporGkg stays flat. windX is meters per second toward the east, windZ toward the south.
export function syntheticColumn(slices: number, domainY: number, vaporGkg: number, windX: number, windZ: number): Float32Array {
    const values = new Float32Array(slices * VALUES_PER_SLICE);

    for (let i = 0; i < slices; i++) {
        const height = (i + 0.5) / slices * domainY;
        const temperature = 295 - 0.0065 * height;
        const pressure = 100000 * Math.pow(temperature / 295, 5.256);
        const offset = i * VALUES_PER_SLICE;
        values[offset] = temperature * Math.pow(100000 / pressure, 0.2857); // theta (K)
        values[offset + 1] = vaporGkg;                                       // qv (g/kg)
        values[offset + 2] = windX;                                          // wind x (m/s, toward east)
        values[offset + 3] = windZ;                                          // wind z (m/s, toward south)
        values[offset + 4] = pressure;                                       // Pa
        values[offset + 5] = temperature;                                    // K
        values[offset + 6] = pressure / (287.05 * temperature);              // density (kg/m^3)
    }

    return values;
}

// Stretch the uneven levels onto the solver's slice centers, (i + 0.5) / slices * domainY.
export function atmosphereColumn(points: SoundingPoint[], slices: number, domainY: number): Float32Array {
    const values = new Float32Array(slices * VALUES_PER_SLICE);

    for (let i = 0; i < slices; i++) {
        const sample = sampleColumn(points, (i + 0.5) / slices * domainY);
        const offset = i * VALUES_PER_SLICE;
        values[offset] = sample.theta;          // K
        values[offset + 1] = sample.vapor;       // qv (g/kg)
        values[offset + 2] = sample.windX;       // m/s toward the east
        values[offset + 3] = sample.windZ;       // m/s toward the south
        values[offset + 4] = sample.pressure;    // Pa
        values[offset + 5] = sample.temperature; // K
        values[offset + 6] = sample.density;     // kg/m^3
    }

    return values;
}
