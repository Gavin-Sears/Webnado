import { GUI } from "lil-gui";

import { device } from "../renderer";
import { constants } from "../shaders/shaders";
import type { Site } from "../stage/stage";
import {
    atmosphereColumn,
    hourlyNames,
    pointsFromHour,
    syntheticColumn,
    VALUES_PER_SLICE,
    type OpenMeteoForecast
} from "./sounding";

// Uploads the sounding column into atmosphereBuffer, which the solver reads as background air.
// Past dates use https://historical-forecast-api.open-meteo.com/v1/forecast
// Today and later use https://api.open-meteo.com/v1/gfs with models=ncep_hrrr_conus
// The GUI hour is UTC. timezone=auto put a May afternoon in Iowa on standard time,
// an hour off daylight time, so the request does not ask Open-Meteo to shift the clock.
export class WeatherData {
    atmosphereBuffer: GPUBuffer;
    date = "2024-05-21";
    hour = 21;
    status = "Standard atmosphere, until a sounding loads.";

    private loading = false;
    private refreshStatus = () => {};
    // Stage points this at the solver. A new column clears the cloud left by the previous one.
    onApplied = () => {};

    constructor(readonly site: Site) {
        this.atmosphereBuffer = device.createBuffer({
            label: "atmosphere",
            size: constants.atmosphereSlices * VALUES_PER_SLICE * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.setAtmosphere(this.placeholderAtmosphere());
    }

    setAtmosphere(values: Float32Array) {
        device.queue.writeBuffer(this.atmosphereBuffer, 0, values);
    }

    // Used until load() succeeds. A failed fetch leaves this in place.
    private placeholderAtmosphere(): Float32Array {
        const values = new Float32Array(constants.atmosphereSlices * VALUES_PER_SLICE);
        for (let i = 0; i < constants.atmosphereSlices; ++i) {
            const height = (i + 0.5) / constants.atmosphereSlices * constants.domainY;
            const temperature = 295 - 0.0065 * height;
            const pressure = 100000 * Math.pow(temperature / 295, 5.256);
            const offset = i * VALUES_PER_SLICE;
            values[offset] = temperature * Math.pow(100000 / pressure, 0.2857);
            values[offset + 1] = 10 * Math.exp(-height / 2500);
            values[offset + 2] = 0;
            values[offset + 3] = 0;
            values[offset + 4] = pressure;
            values[offset + 5] = temperature;
            values[offset + 6] = pressure / (287.05 * temperature);
        }
        return values;
    }

    buildGui(gui: GUI) {
        const folder = gui.addFolder("Weather");
        folder.add(this.site, "latitude", -90, 90, 0.01).name("Latitude");
        folder.add(this.site, "longitude", -180, 180, 0.01).name("Longitude");
        folder.add(this, "date").name("Date");
        folder.add(this, "hour", 0, 23, 1).name("Hour (UTC)");
        folder.add(this, "load").name("Load sounding");
        folder.add(this, "useDryCalm").name("Dry, no wind");
        folder.add(this, "useEastWind").name("Dry, toward east");
        folder.add(this, "useWestWind").name("Dry, toward west");
        const status = folder.add(this, "status").name("Status").disable();
        this.refreshStatus = () => status.updateDisplay();
    }

    // Replace the column and clear the box. Real hours stay humid, so cloud can grow again afterwards.
    async load(): Promise<void> {
        if (this.loading) {
            return;
        }

        const date = this.date.trim();
        const hour = Math.min(23, Math.max(0, Math.round(this.hour)));
        this.hour = hour;
        if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
            this.status = "Date must be YYYY-MM-DD.";
            this.refreshStatus();
            return;
        }

        this.loading = true;
        this.status = "Loading the sounding…";
        this.refreshStatus();

        try {
            const response = await fetch(this.forecastUrl(date));
            if (!response.ok) {
                throw new Error(`Open-Meteo returned HTTP ${response.status}.`);
            }

            const forecast = await response.json() as OpenMeteoForecast;
            const points = pointsFromHour(forecast, date, hour);
            if (points.length < 2) {
                throw new Error(`No sounding for ${date} ${String(hour).padStart(2, "0")}:00 UTC.`);
            }

            const ground = points[0];
            const celsius = ground.temperature - 273.15;
            const speed = Math.hypot(ground.windX, ground.windZ);
            this.applyColumn(
                atmosphereColumn(points, constants.atmosphereSlices, constants.domainY),
                `${date} ${String(hour).padStart(2, "0")}:00 UTC · ${celsius.toFixed(1)} °C · vapor ${ground.vapor.toFixed(1)} g/kg · wind ${speed.toFixed(1)} m/s`
            );
        } catch (error) {
            this.status = error instanceof Error ? error.message : "The sounding could not be loaded.";
        } finally {
            this.loading = false;
            this.refreshStatus();
        }
    }

    // One gram of vapor per kilogram of air. That stays below saturation, so the box should stay clear.
    useDryCalm() {
        this.applySynthetic(1, 0, 0, "Synthetic · dry · no wind");
    }

    // Air moves toward +X. The ground turns part of that wind up and down.
    useEastWind() {
        this.applySynthetic(1, 20, 0, "Synthetic · dry · 20 m/s toward the east");
    }

    useWestWind() {
        this.applySynthetic(1, -20, 0, "Synthetic · dry · 20 m/s toward the west");
    }

    private applySynthetic(vapor: number, windX: number, windZ: number, status: string) {
        this.applyColumn(
            syntheticColumn(constants.atmosphereSlices, constants.domainY, vapor, windX, windZ),
            status
        );
    }

    private applyColumn(values: Float32Array, status: string) {
        this.setAtmosphere(values);
        this.status = status;
        this.refreshStatus();
        this.onApplied();
    }

    // Historical archive for a finished day, live HRRR otherwise. wind_speed_unit=ms on both.
    private forecastUrl(date: string): string {
        const today = new Date().toISOString().slice(0, 10);
        const params = new URLSearchParams({
            latitude: String(this.site.latitude),
            longitude: String(this.site.longitude),
            hourly: hourlyNames().join(","),
            wind_speed_unit: "ms",
            timezone: "UTC"
        });

        if (date >= today) {
            params.set("models", "ncep_hrrr_conus");
            return `https://api.open-meteo.com/v1/gfs?${params}`;
        }

        params.set("start_date", date);
        params.set("end_date", date);
        return `https://historical-forecast-api.open-meteo.com/v1/forecast?${params}`;
    }
}
