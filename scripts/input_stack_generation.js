var geometry  = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq('ADM1_NAME', 'Goa'))
Map.addLayer(geometry)



// Load Sentinel-2 Image Collection and filter by date, cloud coverage, and region
var collection = ee.ImageCollection("COPERNICUS/S2")
                    .filter(ee.Filter.date('2020-01-01', '2021-01-01'))
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                    .filterBounds(geometry);

// Define time steps for seasons
var timeSteps = [
  {'start': '2020-01-01', 'end': '2020-05-01'}, // Winter/Spring
  {'start': '2020-05-02', 'end': '2020-09-01'}, // Summer
  {'start': '2020-09-02', 'end': '2021-01-01'}  // Fall/Winter
];

// Function to get median image for a given time step
var getMedianImage = function(timeStep) {
  return collection.filter(ee.Filter.date(timeStep.start, timeStep.end)).median();
};

// Get median images for each time step
var imageList = timeSteps.map(getMedianImage);

// Select Sentinel-2 bands ≤ 40 meters
// Only include 10m or 20m bands for this example

var bandsToSelect = ['B2', 'B3', 'B4', 'B5','B6','B7', 'B8', 'B8A', 'B11' ,'B12']; 

// Rename the bands for each time step
var renamedImages = imageList.map(function(image, index) {
  var suffix = '_T' + (index + 1); // Add suffix for each time step
  var newBandNames = bandsToSelect.map(function(band) {
    return band + suffix;
  });
  return image.select(bandsToSelect).rename(newBandNames);
});

// Merge all seasonal images into one stack
var mergedStack = ee.Image.cat(renamedImages);

// Visualization parameters for RGB visualization
var visParams = { bands: ['B4_T1', 'B3_T1', 'B2_T1'], min: 0, max: 3000 };

// Add merged stack to the map for visualization
Map.addLayer(mergedStack, visParams, 'Merged Seasonal Stack');
// Map.centerObject(image);

print(mergedStack.bandNames())

// Export the merged image to Google Drive
Export.image.toDrive({
  image: mergedStack.clip(geometry),
  description: 'input_stack_s2',
  folder: 'ee-exports', // Specify your Google Drive folder name here
  fileNamePrefix: 'input_stack',
  region: geometry,
  scale: 40,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});
