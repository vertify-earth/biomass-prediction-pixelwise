# Satellite Imagery Band Structure

This document provides a detailed explanation of the band structure used in the satellite imagery for biomass prediction.

## Overview

The model uses multi-temporal, multi-sensor satellite data from:
- Sentinel-2 optical data (3 time periods)
- Landsat-8 optical data (3 time periods)
- Sentinel-1 SAR data (3 time periods)
- PALSAR-2 L-band SAR data
- Digital Elevation Model (DEM) derivatives

## Detailed Band Structure

The satellite imagery files contain the following bands, in order:

| Band # | Name        | Description                           | Sensor      | Time |
|--------|-------------|---------------------------------------|-------------|------|
| 1      | B2_T1       | Blue                                  | Sentinel-2  | T1   |
| 2      | B3_T1       | Green                                 | Sentinel-2  | T1   |
| 3      | B4_T1       | Red                                   | Sentinel-2  | T1   |
| 4      | B5_T1       | Red Edge 1 (705 nm)                   | Sentinel-2  | T1   |
| 5      | B6_T1       | Red Edge 2 (740 nm)                   | Sentinel-2  | T1   |
| 6      | B7_T1       | Red Edge 3 (783 nm)                   | Sentinel-2  | T1   |
| 7      | B8_T1       | NIR                                   | Sentinel-2  | T1   |
| 8      | B8A_T1      | Narrow NIR                            | Sentinel-2  | T1   |
| 9      | B11_T1      | SWIR 1                                | Sentinel-2  | T1   |
| 10     | B12_T1      | SWIR 2                                | Sentinel-2  | T1   |
| 11     | B2_T2       | Blue                                  | Sentinel-2  | T2   |
| 12     | B3_T2       | Green                                 | Sentinel-2  | T2   |
| 13     | B4_T2       | Red                                   | Sentinel-2  | T2   |
| 14     | B5_T2       | Red Edge 1 (705 nm)                   | Sentinel-2  | T2   |
| 15     | B6_T2       | Red Edge 2 (740 nm)                   | Sentinel-2  | T2   |
| 16     | B7_T2       | Red Edge 3 (783 nm)                   | Sentinel-2  | T2   |
| 17     | B8_T2       | NIR                                   | Sentinel-2  | T2   |
| 18     | B8A_T2      | Narrow NIR                            | Sentinel-2  | T2   |
| 19     | B11_T2      | SWIR 1                                | Sentinel-2  | T2   |
| 20     | B12_T2      | SWIR 2                                | Sentinel-2  | T2   |
| 21     | B2_T3       | Blue                                  | Sentinel-2  | T3   |
| 22     | B3_T3       | Green                                 | Sentinel-2  | T3   |
| 23     | B4_T3       | Red                                   | Sentinel-2  | T3   |
| 24     | B5_T3       | Red Edge 1 (705 nm)                   | Sentinel-2  | T3   |
| 25     | B6_T3       | Red Edge 2 (740 nm)                   | Sentinel-2  | T3   |
| 26     | B7_T3       | Red Edge 3 (783 nm)                   | Sentinel-2  | T3   |
| 27     | B8_T3       | NIR                                   | Sentinel-2  | T3   |
| 28     | B8A_T3      | Narrow NIR                            | Sentinel-2  | T3   |
| 29     | B11_T3      | SWIR 1                                | Sentinel-2  | T3   |
| 30     | B12_T3      | SWIR 2                                | Sentinel-2  | T3   |
| 31     | SR_B2_T1    | Blue                                  | Landsat-8   | T1   |
| 32     | SR_B3_T1    | Green                                 | Landsat-8   | T1   |
| 33     | SR_B4_T1    | Red                                   | Landsat-8   | T1   |
| 34     | SR_B5_T1    | NIR                                   | Landsat-8   | T1   |
| 35     | SR_B6_T1    | SWIR 1                                | Landsat-8   | T1   |
| 36     | SR_B7_T1    | SWIR 2                                | Landsat-8   | T1   |
| 37     | SR_B2_T2    | Blue                                  | Landsat-8   | T2   |
| 38     | SR_B3_T2    | Green                                 | Landsat-8   | T2   |
| 39     | SR_B4_T2    | Red                                   | Landsat-8   | T2   |
| 40     | SR_B5_T2    | NIR                                   | Landsat-8   | T2   |
| 41     | SR_B6_T2    | SWIR 1                                | Landsat-8   | T2   |
| 42     | SR_B7_T2    | SWIR 2                                | Landsat-8   | T2   |
| 43     | SR_B2_T3    | Blue                                  | Landsat-8   | T3   |
| 44     | SR_B3_T3    | Green                                 | Landsat-8   | T3   |
| 45     | SR_B4_T3    | Red                                   | Landsat-8   | T3   |
| 46     | SR_B5_T3    | NIR                                   | Landsat-8   | T3   |
| 47     | SR_B6_T3    | SWIR 1                                | Landsat-8   | T3   |
| 48     | SR_B7_T3    | SWIR 2                                | Landsat-8   | T3   |
| 49     | VV_T1       | VV Polarization                       | Sentinel-1  | T1   |
| 50     | VH_T1       | VH Polarization                       | Sentinel-1  | T1   |
| 51     | VV_T2       | VV Polarization                       | Sentinel-1  | T2   |
| 52     | VH_T2       | VH Polarization                       | Sentinel-1  | T2   |
| 53     | VV_T3       | VV Polarization                       | Sentinel-1  | T3   |
| 54     | VH_T3       | VH Polarization                       | Sentinel-1  | T3   |
| 55     | HH          | HH Polarization                       | PALSAR-2    | -    |
| 56     | HV          | HV Polarization                       | PALSAR-2    | -    |
| 57     | elevation   | Elevation                             | SRTM DEM    | -    |
| 58     | slope       | Slope                                 | Derived     | -    |
| 59     | aspect      | Aspect                                | Derived     | -    |
<!-- | 60     | NDVI_T1     | Normalized Difference Vegetation Index| Calculated  | T1   |
| 61     | NDVI_T2     | Normalized Difference Vegetation Index| Calculated  | T2   |
| 62     | NDVI_T3     | Normalized Difference Vegetation Index| Calculated  | T3   |
| 63     | EVI_T1      | Enhanced Vegetation Index            | Calculated  | T1   |
| 64     | EVI_T2      | Enhanced Vegetation Index            | Calculated  | T2   |
| 65     | EVI_T3      | Enhanced Vegetation Index            | Calculated  | T3   |
| 66     | SAVI_T1     | Soil Adjusted Vegetation Index       | Calculated  | T1   |
| 67     | SAVI_T2     | Soil Adjusted Vegetation Index       | Calculated  | T2   |
| 68     | SAVI_T3     | Soil Adjusted Vegetation Index       | Calculated  | T3   |
| 69     | MSAVI2_T1   | Modified Soil Adjusted Veg. Index    | Calculated  | T1   |
| 70     | MSAVI2_T2   | Modified Soil Adjusted Veg. Index    | Calculated  | T2   |
| 71     | MSAVI2_T3   | Modified Soil Adjusted Veg. Index    | Calculated  | T3   |
| 72     | NDMI_T1     | Normalized Difference Moisture Index  | Calculated  | T1   |
| 73     | NDMI_T2     | Normalized Difference Moisture Index  | Calculated  | T2   |
| 74     | NDMI_T3     | Normalized Difference Moisture Index  | Calculated  | T3   |
| 75     | NBR_T1      | Normalized Burn Ratio                | Calculated  | T1   |
| 76     | NBR_T2      | Normalized Burn Ratio                | Calculated  | T2   |
| 77     | NBR_T3      | Normalized Burn Ratio                | Calculated  | T3   |
| 78     | NDWI_T1     | Normalized Difference Water Index    | Calculated  | T1   |
| 79     | NDWI_T2     | Normalized Difference Water Index    | Calculated  | T2   |
| 80     | NDWI_T3     | Normalized Difference Water Index    | Calculated  | T3   |
| 81-83  | LAI_T1-T3   | Leaf Area Index                      | Calculated  | T1-T3 |
| 84-86  | BSI_T1-T3   | Bare Soil Index                      | Calculated  | T1-T3 |
| 87-89  | NDBI_T1-T3  | Normalized Difference Built-up Index | Calculated  | T1-T3 |
| 90-92  | NDRE_T1-T3  | Normalized Diff. Red Edge Index      | Calculated  | T1-T3 |
| 93-95  | S1_ratio_T1-T3 | VV/VH ratio                        | Calculated  | T1-T3 |
| 96-98  | LS_ratio_T1-T3 | NIR/SWIR ratio from Landsat        | Calculated  | T1-T3 |
| 99     | Texture GLCM| Gray Level Co-occurrence Matrix       | Calculated  | -    | -->

## Time Periods

The multi-temporal data represents three time periods:
- T1: Early dry season (typically November-January)
- T2: Late dry season (typically February-April)
- T3: Wet season (typically June-September)

This multi-temporal approach captures seasonal variations in vegetation phenology which improves biomass estimation accuracy.

## Pre-processing

The satellite data undergoes several pre-processing steps:
1. **Atmospheric correction**: Converting to surface reflectance
2. **Cloud masking**: Removing cloud and cloud shadow pixels
3. **Speckle filtering**: Reducing noise in SAR data
4. **Terrain correction**: Correcting for topographic effects
5. **Co-registration**: Ensuring all imagery is properly aligned
6. **Resampling**: Converting all data to a common resolution (10m)

## Data Preparation Guidelines

When preparing your own data:

1. **Temporal consistency**: Try to match the seasonal pattern (early dry, late dry, wet season)
2. **Complete bands**: Include as many of the bands as possible for best results
3. **Cloud-free images**: Select images with minimal cloud cover
4. **Correct preprocessing**: Apply atmospheric correction, terrain correction, and noise reduction
5. **Pixel alignment**: Ensure all bands are perfectly co-registered

## Special Considerations

- **Missing data**: The model is designed to handle missing bands, but performance will be best with complete data
- **Radiometric quality**: Use images with good radiometric quality (avoid haze, smoke, clouds)
- **Temporal gap**: Keep the temporal gap between imagery and ground truth biomass measurements as small as possible
- **Seasonal effects**: Be aware that seasonality affects biomass estimation accuracy