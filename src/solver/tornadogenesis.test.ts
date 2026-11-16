import { device, initWebGPU } from '../renderer';
import { domainName } from '../shaders/shaders';
import { Fields } from '../stage/fields';
import { WeatherData } from '../weather/noaa_loader';
import { ObstacleGrid } from '../gis/obstacle_grid';
import { FluidSolver } from './fluid_solver';
import { StormDiagnostics, type StormReport } from './storm_diagnostics';

const output = document.getElementById('results')!;
const reports: StormReport[] = [];
Object.assign(window, { stormReports: reports });

async function run() {
    if (domainName !== 'supercell' && domainName !== 'storm-focus') {
        throw new Error('Use domain=supercell for the full-depth experiment.');
    }
    await initWebGPU();
    device.pushErrorScope('validation');
    const fields = new Fields(false);
    fields.setObstacles(new ObstacleGrid().materials);
    const weather = new WeatherData({ latitude: 35.5, longitude: -98 });
    const solver = new FluidSolver(fields, weather);
    const diagnostics = new StormDiagnostics();
    solver.startElRenoExperiment();
    const query = new URLSearchParams(location.search);
    solver.dt = Number(query.get('dt') ?? 1);
    solver.multigridCycles = Number(query.get('cycles') ?? 4);
    solver.confinement = Number(query.get('confinement') ?? 0);
    solver.storm.liftSpeed = Number(query.get('lift') ?? 10);
    solver.storm.liftDuration = Number(query.get('duration') ?? 1200);
    solver.mixing = Number(query.get('mixing') ?? 0.1);
    solver.turbulence = query.get('turbulence') !== 'off';
    solver.storm.highOrder = query.get('advection') !== 'first';
    const equilibrium = query.get('control') === 'equilibrium';
    if (equilibrium) {
        solver.storm.liftSpeed = 0;
        solver.turbulence = false;
        solver.confinement = 0;
    }
    const steps = Number(query.get('steps') ?? 3600);
    const reportEvery = Number(query.get('reportEvery') ?? 100);

    try {
        for (let step = 1; step <= steps; step++) {
            const encoder = device.createCommandEncoder();
            solver.step(encoder);
            device.queue.submit([encoder.finish()]);
            if (step % reportEvery === 0 || step === steps) {
                const report = await diagnostics.measure(fields, solver.storm.elapsed);
                solver.storm.track(report);
                reports.push(report);
                output.textContent = JSON.stringify(reports, null, 2);
                console.log('STORM', JSON.stringify(report));
                if (report.invalid || report.speed > 250 || report.temperature > 100) {
                    throw new Error('The atmospheric experiment exceeded its numerical bounds.');
                }
                if (report.divergenceRms > 5e-4) {
                    throw new Error('Pressure projection exceeded the interior divergence target.');
                }
                if (equilibrium && (report.updraft > 0.001 || report.temperature > 0.001
                    || report.interiorVerticalVorticity > 1e-6)) {
                    throw new Error('The unperturbed atmosphere developed artificial motion.');
                }
            } else if (step % 10 === 0) {
                await device.queue.onSubmittedWorkDone();
            }
        }
        const error = await device.popErrorScope();
        if (error) {
            throw new Error(error.message);
        }
        output.textContent += '\nExperiment completed. These diagnostics do not classify a tornado.';
    } finally {
        diagnostics.destroy();
        solver.destroy();
        weather.atmosphereBuffer.destroy();
    }
}

void run().catch(error => {
    output.textContent += `\nFAIL ${error.message}`;
    console.error(error);
});
