# Biomass Data Directory

This directory should contain your ground truth biomass measurements in GeoTIFF format.

## Expected Files

Place your biomass data files here with the following naming convention:
```
site1_biomass.tif
site2_biomass.tif
...
```

## File Requirements

- Format: GeoTIFF (.tif or .tiff)
- Values: Above-ground biomass in Mg/ha (megagrams per hectare)
- Single band: Each file should contain a single band
- Co-registered: Must be perfectly aligned with the corresponding satellite image
- Valid values: Positive values (negative or zero values will be masked out)

## Data Sources

Biomass ground truth data can be obtained from:
- Field surveys with plot-based measurements
- LiDAR-derived biomass maps
- Existing biomass products from research institutions
- Published biomass maps from peer-reviewed studies

## Pre-processing Recommendations

- Ensure data is in Mg/ha units (convert if necessary)
- Reproject to match the satellite imagery projection
- Resample to match the satellite imagery resolution
- Remove or flag outliers (values > 1000 Mg/ha are often considered outliers)
- Fill small data gaps if appropriate using interpolation techniques