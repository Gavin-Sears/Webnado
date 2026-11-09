import { GUI } from "lil-gui";

import { device } from "../renderer";
import { constants } from "../shaders/shaders";
import type { Site } from "../stage/stage";

const VALUES_PER_SLICE = 8;

// TODO:
// Input: a Site and fetched data
// Output: atmosphereBuffer for the solver, as array<AtmosphereSample> (common.wgsl).
//   constants.atmosphereSlices slices, evenly spaced vertical slices
//   Per slice: theta (K), qv (g/kg), wind x (m/s), wind z (m/s), pressure (Pa), temperature (K), density (kg/m^3)
//
// Source: Open-Meteo (free and doesn't require a key), serving NOAA's GFS and HRRR models. Docs: https://open-meteo.com/en/docs
//   Live:  https://api.open-meteo.com/v1/gfs with models=ncep_hrrr_conus
//   Past:  https://historical-forecast-api.open-meteo.com/v1/forecast with start_date and end_date
//   Request wind_speed_unit=ms and hourly temperature_, relative_humidity_, wind_speed_, wind_direction_ and
//   geopotential_height_ + {1000, 975, 950, 925, 900, 850, 800, 700, 600, 500, 400}hPa, plus temperature_2m,
//   relative_humidity_2m, surface_pressure, wind_speed_10m and wind_direction_10m.
//   Responses are in UTC (SO MAKE SURE TO CONVERT TO LOCAL TIME FOR DISPLAY!), °C and hPa, and include the ground's elevation (m).
export class WeatherData {
    atmosphereBuffer: GPUBuffer;

    constructor(readonly site: Site) {
        this.atmosphereBuffer = device.createBuffer({
            label: "atmosphere",
            size: constants.atmosphereSlices * VALUES_PER_SLICE * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.setAtmosphere(this.placeholderAtmosphere());
    }

    setAtmosphere(values: Float32Array) {
        device.queue.writeBuffer(this.atmosphereBuffer, 0, values);
    }

    // TODO: Placeholder for weather data. This is just some air that cools as it gets higher. 
    // Replace this with the fetched data.
    private placeholderAtmosphere(): Float32Array {
        const values = new Float32Array(constants.atmosphereSlices * VALUES_PER_SLICE);
        for (let i = 0; i < constants.atmosphereSlices; ++i) {
            const height = (i + 0.5) / constants.atmosphereSlices * constants.domainY;
            const temperature = 295 - 0.0065 * height;
            const pressure = 100000 * Math.pow(temperature / 295, 5.256);
            const offset = i * VALUES_PER_SLICE;
            values[offset] = temperature * Math.pow(100000 / pressure, 0.2857);     // theta
            values[offset + 1] = 10 * Math.exp(-height / 2500);                     // qv
            values[offset + 2] = 0;                                                 // wind x
            values[offset + 3] = 0;                                                 // wind z
            values[offset + 4] = pressure;
            values[offset + 5] = temperature;
            values[offset + 6] = pressure / (287.05 * temperature);                 // density
        }
        return values;
    }

    // TODO: Make weather loading GUI (location, date, hour, load button, etc.)
    buildGui(_gui: GUI) {
    }
}
