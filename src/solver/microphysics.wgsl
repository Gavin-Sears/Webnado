// Warm-rain microphysics for the four thermo channels.
// Kessler autoconversion and accretion (Kessler 1969, as used by Klemp and
// Wilhelmson 1978): cloud forms or evaporates until the vapor is at saturation,
// then cloud becomes rain, then rain collects cloud. Rain evaporates only while
// the air is still short of saturation, and never past it.
//
// theta' and qv' stay relative to the sounding. qc and qr stay absolute, in g/kg.
// The work below is in kg/kg and is converted back at the store.

const latentHeat = 2.5e6;
const specificHeat = 1004.0;
const vaporGas = 461.5;
const referencePressure = 1.0e5;
const poissonKappa = 0.2857;
const autoconversionRate = 0.001;
const autoconversionThreshold = 0.001;
const accretionRate = 2.2;

// Bolton saturation vapor pressure, then qs = 0.622 * es / (p - es), in kg/kg.
fn saturationMixingRatio(temperature: f32, pressure: f32) -> f32 {
    let celsius = temperature - 273.15;
    let es = 611.2 * exp(17.67 * celsius / max(celsius + 243.5, 1.0));
    return 0.622 * es / max(pressure - es, 1.0);
}

// Latent heat puts a fraction of a vapor surplus back into temperature, so not all of it condenses.
fn saturationFactor(qs: f32, temperature: f32) -> f32 {
    return 1.0 + qs * latentHeat * latentHeat / (specificHeat * vaporGas * temperature * temperature);
}

// Kessler fall speed in m/s. +Y is up, so callers move rain toward smaller y.
fn terminalFallSpeed(rain: f32, density: f32) -> f32 {
    if (rain <= 1e-8) {
        return 0.0;
    }

    // 0.001 takes kg/m^3 * kg/kg into the g/cm^3 form of Kessler's fall speed.
    let water = max(density, 0.3) * rain * 0.001;
    return 36.34 * pow(water, 0.1364) * sqrt(1.2 / max(density, 0.3));
}

// After advection. Reads theta', qv', qc, qr plus the sounding and writes the same four channels.
// Buoyancy uses the result on the next step, because forces already ran this step.
@compute @workgroup_size(8, 8, 4)
fn microphysics(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var stored = vec4f(0);

    if (fluid(cell)) {
        let sample = textureLoad(thermoIn, cell, 0);
        let air = background(f32(cell.y) + 0.5);
        var theta = sample.x + air.state.x;
        var vapor = max(sample.y + air.state.y, 0.0) * 0.001;
        var cloud = max(sample.z, 0.0) * 0.001;
        var rain = max(sample.w, 0.0) * 0.001;
        let pressure = air.air.x;
        var temperature = theta * pow(pressure / referencePressure, poissonKappa);

        // A bad sounding sample is copied through. Condensation on it would write NaNs into the grid.
        if (theta > 100.0 && temperature > 150.0 && pressure > 1000.0) {
            let qs = saturationMixingRatio(temperature, pressure);
            let factor = saturationFactor(qs, temperature);
            var exchange = (vapor - qs) / factor;
            exchange = clamp(exchange, -cloud, vapor);
            vapor -= exchange;
            cloud += exchange;
            theta += (latentHeat / specificHeat) * (theta / temperature) * exchange;
            temperature = theta * pow(pressure / referencePressure, poissonKappa);

            let converted = min(cloud, autoconversionRate * max(cloud - autoconversionThreshold, 0.0) * parameters.dt);
            cloud -= converted;
            rain += converted;

            if (cloud > 0.0 && rain > 0.0) {
                let collected = min(cloud, accretionRate * cloud * pow(rain, 0.875) * parameters.dt);
                cloud -= collected;
                rain += collected;
            }

            // Klemp and Wilhelmson evaporation, in Pa and kg/kg, stopped at saturation.
            // Their original constants are in millibars and g/cm^3, so they are not passed raw SI values.
            let qsAfter = saturationMixingRatio(temperature, pressure);
            let room = max(qsAfter - vapor, 0.0) / saturationFactor(qsAfter, temperature);
            // One adjustment does not land exactly on saturation, so rain waits until the cloud is gone.
            if (room > 0.0 && rain > 0.0 && cloud <= 1e-8) {
                let density = max(air.air.z, 0.3);
                let water = density * rain * 0.001;
                let vent = (1.6 + 124.9 * pow(water, 0.2046)) * pow(water, 0.525);
                let denom = 2.55e8 / max(pressure * qsAfter, 1e-6) + 5.4e5;
                let rate = vent / denom * max(qsAfter - vapor, 0.0) / max(0.001 * density * qsAfter, 1e-12);
                let evaporated = min(rain, min(room, max(rate, 0.0) * parameters.dt));
                vapor += evaporated;
                rain -= evaporated;
                theta -= (latentHeat / specificHeat) * (theta / temperature) * evaporated;
            }

            stored = vec4f(
                theta - air.state.x,
                vapor * 1000.0 - air.state.y,
                cloud * 1000.0,
                rain * 1000.0
            );
            stored.y = max(stored.y, -air.state.y);
            stored.z = max(stored.z, 0.0);
            stored.w = max(stored.w, 0.0);
        } else {
            stored = sample;
        }
    }

    textureStore(thermoOut, cell, stored);
}

// Donor-cell fall of qr. Separate pass: a storage texture cannot read a neighbor this pass just wrote.
// The atmospheric experiment drains precipitation when it reaches a solid surface.
@compute @workgroup_size(8, 8, 4)
fn sedimentRain(@builtin(global_invocation_id) id: vec3u) {
    let cell = vec3i(id);
    if (!inside(cell)) {
        return;
    }

    var sample = vec4f(0);

    if (fluid(cell)) {
        sample = textureLoad(thermoIn, cell, 0);
        let rainHere = max(sample.w, 0.0) * 0.001;
        let air = background(f32(cell.y) + 0.5);
        let speed = terminalFallSpeed(rainHere, air.air.z);
        let courant = clamp(speed * parameters.dt / cellSize.y, 0.0, 1.0);
        let below = cell - vec3i(0, 1, 0);
        let fallsOut = !inside(below) || fluid(below) || parameters.rainDrain > 0.0;
        let leaving = select(0.0, courant * rainHere, fallsOut);

        var incoming = 0.0;
        let above = cell + vec3i(0, 1, 0);
        if (fluid(above)) {
            let rainAbove = max(textureLoad(thermoIn, above, 0).w, 0.0) * 0.001;
            let airAbove = background(f32(above.y) + 0.5);
            let speedAbove = terminalFallSpeed(rainAbove, airAbove.air.z);
            let courantAbove = clamp(speedAbove * parameters.dt / cellSize.y, 0.0, 1.0);
            incoming = courantAbove * rainAbove * airAbove.air.z / max(air.air.z, 1e-6);
        }

        sample.w = max(rainHere - leaving + incoming, 0.0) * 1000.0;
    }

    textureStore(thermoOut, cell, sample);
}
