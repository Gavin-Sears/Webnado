import { GisTerrain } from "../gis/gis_terrain";
import { FluidSolver } from "../solver/fluid_solver";
import { WeatherData } from "../weather/noaa_loader";
import { Camera } from "./camera";
import { Fields } from "./fields";

// The center of the simulated box on Earth, shared by weather and tiles
export interface Site {
    latitude: number;
    longitude: number;
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
        readonly solver: FluidSolver
    ) {}

    static async create(): Promise<Stage> {
        const site = DEFAULT_SITE;
        const camera = new Camera();
        const fields = new Fields();

        const weather = new WeatherData(site);
        const gis = await GisTerrain.create(site);
        fields.setObstacles(gis.obstacleMaterials());

        const solver = new FluidSolver(fields, weather);
        weather.onApplied = () => solver.resetAir();
        await weather.load();

        return new Stage(site, camera, fields, weather, gis, solver);
    }
}
