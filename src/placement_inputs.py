import argparse
import json

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from .config import output_dir, site_config
from .placement import hand

WET_THRESHOLD = 0.1
RIVER_BUFFER_M = 20
SLOPE_PERCENTILE = 40
URBAN_DILATION_M = 200

DRIVE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
         "residential", "service", "living_street", "road",
         "motorway_link", "trunk_link", "primary_link",
         "secondary_link", "tertiary_link"}
WATERWAY = {"river", "stream", "canal"}

LAYER_NAMES = ("dem", "slope", "river_mask", "hand", "dist_to_river",
               "road_mask", "building_mask", "water_mask", "cctv_candidate",
               "cctv_step", "road_available", "ever_wet", "flood_freq_event",
               "flood_freq_duration", "floodnet_score")


def river_mask(site):
    from .data import load_static

    sc = site_config(site)
    if sc.get("river_mask"):
        return np.load(sc["river_mask"]) > 0
    _, _, q_pathway, _ = load_static(site)
    return q_pathway > 0


def flood_layers(site, train_events, thr=WET_THRESHOLD):
    from .data import load_event

    ever_wet = None
    duration = None
    for event in train_events:
        _, depth = load_event(site, event)
        wet_count = (depth > thr).sum(axis=0)
        if ever_wet is None:
            ever_wet = np.zeros(wet_count.shape, np.int32)
            duration = np.zeros(wet_count.shape, np.float64)
        ever_wet += (wet_count > 0)
        duration += wet_count / depth.shape[0]
        del depth

    n = len(train_events)
    freq_event = (ever_wet / n).astype(np.float32)
    freq_duration = (duration / n).astype(np.float32)
    score = (freq_event + 1e-3 * freq_duration).astype(np.float32)
    return ever_wet, freq_event, freq_duration, score


def osm_layers(site, shape):
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    sc = site_config(site)
    H, W = shape
    dx = sc["dx"]
    xmin, ymax = sc["origin"]
    xmax, ymin = xmin + W * dx, ymax - H * dx
    transform = from_origin(xmin, ymax, dx, dx)
    crs = sc["crs"]
    half = sc["road_halfwidth"]

    lines = gpd.read_file(sc["osm"], layer="lines").to_crs(crs)
    bounds = lines.total_bounds
    assert (bounds[0] < xmax and bounds[2] > xmin
            and bounds[1] < ymax and bounds[3] > ymin), "OSM does not overlap domain"

    roads = lines[lines["highway"].notna()]
    roads = roads[roads["highway"].isin(DRIVE)].cx[xmin:xmax, ymin:ymax]
    assert len(roads) > 0, "no roads inside domain"
    road = rasterize([(g.buffer(half), 1) for g in roads.geometry],
                     out_shape=shape, transform=transform, fill=0,
                     dtype=np.uint8, all_touched=True).astype(bool)

    try:
        polys = gpd.read_file(sc["osm"], layer="multipolygons").to_crs(crs)
        buildings = polys[polys["building"].notna()].cx[xmin:xmax, ymin:ymax]
        building = rasterize([(g, 1) for g in buildings.geometry],
                             out_shape=shape, transform=transform, fill=0,
                             dtype=np.uint8, all_touched=True).astype(bool)
        k = int(round(URBAN_DILATION_M / dx)) | 1
        urban = binary_dilation(building, np.ones((k, k)))
    except Exception:
        building = np.zeros(shape, bool)
        urban = np.ones(shape, bool)

    water_extra = np.zeros(shape, bool)
    if sc.get("landuse"):
        try:
            lu = gpd.read_file(sc["landuse"])
            b = lu.total_bounds
            if not (b[0] < xmax and b[2] > xmin and b[1] < ymax and b[3] > ymin):
                lu = lu.to_crs(crs)
            water = lu[lu["Type"] == "water"]
            water_extra = rasterize([(g, 1) for g in water.geometry],
                                    out_shape=shape, transform=transform, fill=0,
                                    dtype=np.uint8, all_touched=True).astype(bool)
        except Exception:
            pass

    return road, building, urban, water_extra


def osm_waterway_iou(site, river, shape):
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    sc = site_config(site)
    H, W = shape
    dx = sc["dx"]
    xmin, ymax = sc["origin"]
    xmax, ymin = xmin + W * dx, ymax - H * dx
    transform = from_origin(xmin, ymax, dx, dx)

    lines = gpd.read_file(sc["osm"], layer="lines").to_crs(sc["crs"])
    if "waterway" not in lines.columns:
        return None
    wl = lines[lines["waterway"].isin(WATERWAY)].cx[xmin:xmax, ymin:ymax]
    if len(wl) == 0:
        return None

    best = 0.0
    for buf in (dx, 2 * dx, 3 * dx, 4 * dx, 6 * dx):
        m = rasterize([(g.buffer(buf), 1) for g in wl.geometry],
                      out_shape=shape, transform=transform, fill=0,
                      dtype=np.uint8, all_touched=True).astype(bool)
        iou = (m & river).sum() / max((m | river).sum(), 1)
        best = max(best, float(iou))
    return best


def build(site):
    from .data import load_folds, load_static

    sc = site_config(site)
    dem, slope, _, _ = load_static(site)
    shape = dem.shape
    river = river_mask(site)
    assert river.any(), "river mask is empty"

    hand_field = hand(dem, river)
    dist = (distance_transform_edt(~river) * sc["dx"]).astype(np.float32)

    road, building, urban, water_extra = osm_layers(site, shape)
    water = river | water_extra
    river_buffer = binary_dilation(
        water, np.ones((3, 3)),
        iterations=max(1, int(RIVER_BUFFER_M // sc["dx"])))

    road_available = road & ~river_buffer
    cctv_step = road_available & urban
    threshold = np.percentile(slope[cctv_step], SLOPE_PERCENTILE)
    cctv_candidate = cctv_step & (slope <= threshold)

    _, fixed = load_folds(site)
    ever_wet, freq_event, freq_duration, score = flood_layers(
        site, fixed["train"])

    layers = dict(dem=dem, slope=slope, river_mask=river.astype(np.uint8),
                  hand=hand_field, dist_to_river=dist,
                  road_mask=road.astype(np.uint8),
                  building_mask=building.astype(np.uint8),
                  water_mask=water.astype(np.uint8),
                  cctv_candidate=cctv_candidate.astype(np.uint8),
                  cctv_step=cctv_step.astype(np.uint8),
                  road_available=road_available.astype(np.uint8),
                  ever_wet=ever_wet, flood_freq_event=freq_event,
                  flood_freq_duration=freq_duration, floodnet_score=score)

    out = output_dir("auxiliary", site)
    for name, arr in layers.items():
        np.save(out / f"{name}.npy", arr)

    flood_extent = ever_wet >= 1
    low_hand = hand_field <= np.percentile(hand_field, 10)
    low_dem = dem <= np.percentile(dem, 10)
    diagnostics = dict(
        floodplain_elevation_sigma=float(dem[(~river) & low_dem].std()),
        random_baseline=float(flood_extent.mean()),
        hand_lowest10_in_flood_extent=float(
            (low_hand & flood_extent).sum() / low_hand.sum()),
        elevation_lowest10_in_flood_extent=float(
            (low_dem & flood_extent).sum() / low_dem.sum()),
        iou_hand_elevation=float((low_hand & low_dem).sum()
                                 / (low_hand | low_dem).sum()),
        n_train_events=len(fixed["train"]),
        river_pixels=int(river.sum()),
    )
    try:
        diagnostics["osm_waterway_iou"] = osm_waterway_iou(site, river, shape)
    except Exception:
        diagnostics["osm_waterway_iou"] = None

    with open(out / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    for k, v in diagnostics.items():
        print(f"  {k:<38} {v}")
    return layers


def load_layers(site):
    d = output_dir("auxiliary", site, create=False)
    if not (d / "hand.npy").exists():
        raise FileNotFoundError(
            f"auxiliary layers missing, run: python -m src.placement_inputs "
            f"--site {site}")
    layers = {}
    for name in LAYER_NAMES:
        arr = np.load(d / f"{name}.npy")
        if name in ("river_mask", "road_mask", "building_mask", "water_mask",
                    "cctv_candidate", "cctv_step", "road_available"):
            arr = arr.astype(bool)
        layers[name] = arr
    return layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    args = ap.parse_args()
    build(args.site)


if __name__ == "__main__":
    main()
