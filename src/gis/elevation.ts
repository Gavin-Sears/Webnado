export async function groundElevation(latitude: number, longitude: number): Promise<number> {
    const query = new URLSearchParams({ latitude: String(latitude), longitude: String(longitude) });
    const response = await fetch(`https://api.open-meteo.com/v1/elevation?${query}`, { signal: AbortSignal.timeout(15000) });
    if (!response.ok) throw new Error(`Elevation lookup returned HTTP ${response.status}`);
    const data = await response.json();
    const height = data.elevation?.[0];
    if (!Number.isFinite(height)) throw new Error('Elevation lookup returned no ground height');
    return height;
}
