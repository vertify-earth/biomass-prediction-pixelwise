# Satellite Data Directory

This directory should contain your multi-band satellite imagery in GeoTIFF format.

## Expected Files

Place your satellite imagery files here with the following naming convention:
```
site1_satellite.tif
site2_satellite.tif
...
```

## File Requirements

- Format: GeoTIFF (.tif or .tiff)
- Bands: See [SATELLITE_BANDS.md](../../docs/SATELLITE_BANDS.md) for the complete band structure
- Resolution: Ideally 10-30m resolution
- Projection: Any standard projection (will be handled by the code)

## Data Sources

Satellite data can be obtained from:
- [Copernicus Open Access Hub](https://scihub.copernicus.eu/) for Sentinel-1 and Sentinel-2
- [USGS Earth Explorer](https://earthexplorer.usgs.gov/) for Landsat-8
- [ALOS PALSAR Mosaic](https://www.eorc.jaxa.jp/ALOS/en/palsar_fnf/fnf_index.htm) for PALSAR data
- [SRTM Data](https://earthexplorer.usgs.gov/) for DEM

## Pre-processing Recommendations

- Apply atmospheric correction to optical data
- Apply speckle filtering to SAR data
- Ensure cloud-free imagery when possible
- Co-register all images to ensure perfect alignment