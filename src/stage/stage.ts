import { GisTerrain } from "../gis/gis_terrain";
import { FluidSolver } from "../solver/fluid_solver";
import { WeatherData } from "../weather/noaa_loader";
import { Camera } from "./camera";
import { Fields } from "./fields";
import { Debris } from "../gis/debris";

// The center of the simulated box on Earth, shared by weather and tiles
export interface Site {
    latitude: number;
    longitude: number;
    elevation?: number;
}

// Default is the location of an EF4 tornado in Greenfield Iowa on May 21, 2024
// This doesn't really mean anything right now, and we might not even use this.
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
        const fields = new Fields();

        const weather = new WeatherData(site);

        const gis = await GisTerrain.create(site);
        await gis.connect(fields);

        const solver = new FluidSolver(fields, weather);
        const debris = new Debris(fields, camera, () => solver.dust.seed());

        return new Stage(site, camera, fields, weather, gis, solver, debris);
    }
}
