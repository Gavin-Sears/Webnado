import { GisTerrain } from "../gis/gis_terrain";
import { FluidSolver } from "../solver/fluid_solver";
import { WeatherData } from "../weather/noaa_loader";
import { Camera } from "./camera";
import { Fields } from "./fields";
import { Debris } from "../gis/debris";
import { constants } from '../shaders/shaders';

// The center of the simulated box on Earth, shared by weather and tiles
export interface Site {
    latitude: number;
    longitude: number;
    elevation?: number;
}

// Greenfield, Iowa. WeatherData loads 21 May 2024, 21:00 UTC from here.
const DEFAULT_SITE: Site = { latitude: 41.31, longitude: -94.46 };

// CHECKITOUT: holds everything that outlives a renderer and connects the four sections
export class Stage {
    private constructor(
        readonly site: Site,
        readonly camera: Camera,
        readonly fields: Fields,
        readonly weather: WeatherData,
        readonly gis: GisTerrain,
        readonly solver: FluidSolver,
        readonly debris: Debris
    ) {}

    static async create(): Promise<Stage> {
        const site = DEFAULT_SITE;
        const camera = new Camera();
        const query = new URLSearchParams(location.search);
        const atmosphericCase = ['elreno', 'hrrr'].includes(query.get('case') ?? '');
        const fields = new Fields(!atmosphericCase);

        const weather = new WeatherData(site);
        if (constants.spatialWeather) {
            const input = query.get('weather');
            if (!input) {
                throw new Error('The HRRR experiment needs a spatial weather manifest URL (?weather=...).');
            }
            await weather.loadSpatial(input);
        }

        const gis = await GisTerrain.create(site);
        await gis.connect(fields);
        if (weather.spatial && !weather.spatialFlat) {
            gis.useSpatialTerrain(weather.spatial);
        } else if (weather.spatial && weather.spatialFlat) {
            gis.status = 'Flat comparison terrain for the weather window';
        }

        const solver = new FluidSolver(fields, weather);
        weather.onApplied = () => solver.reset();
        if (constants.spatialWeather) {
            solver.startSpatialExperiment();
        } else if (atmosphericCase) {
            solver.startElRenoExperiment();
        } else {
            void weather.load();
        }
        const debris = new Debris(fields, camera, position => {
            if (position) {
                solver.dust.seedNear(position);
            } else {
                solver.dust.seed();
            }
        });

        return new Stage(site, camera, fields, weather, gis, solver, debris);
    }
}
