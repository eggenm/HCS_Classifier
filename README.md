# HCS_Classifier

##Overview
This software was originally designed to run a classification returning a map of High Carbon Stock forest over the Indonesian islands of Kalimantan, Sumatra and Papua.
It consists of a few major modules
1. Image Ingestion and Data Collection
2. Data Preparation
3. Model parameterization
4. Classification
5. Post Processing

##Image Ingestion and Data Collection
###Ingest Images from Google Earth to Google Cloud storage, download and mosaic into an island directory

1. Collect lat-long square for grid you would like to ingest data from Google
2. Add lat-long to ingest_by_grid.py, point code to your google cloud storage (change 'bucket' parameter):
   Line 89:  task = ee.batch.Export.image.toCloudStorage(image=myimage, fileNamePrefix =prefix , bucket='hcsa_forest_mapping_training_bucket',  crs='EPSG:4326', scale=30 )
3. Run ingest_by_grid.py which takes median pixel value for a 2 year window of landsat-8 bands or for Sentinel-1 SAR takes the median value of the focal mean for a variety of window sizes.
4. Go to your Google Cloud Storage and download files as desired.
4. Change mosaic_tiles.py for your input and output directories
5. Run mosaic_tiles.py

##Data Preparation
###Create concession-level data inputs from island inputs
1. With all island data downloaded, setup individual concessions for analysis or classification.
2. Data_helper.py has a runnable section at the bottom where you can supply the following:
   a. List of bands to prep (must be named appropriately see: satellite_image_ops.py for constants for band names by satelite platform. )
   b. List of concessions to ingest (must be named as named on filesystem) or reference to a shapefile of a study area.
   c. List of years to ingest
3. Modify these input parameters as needed, run the class and rasters for the study areas will be output in the appropriate directories. (directory structure must exist)

###Create fixed class inputs from island inputs
1. Starting with a set of .kmz files in the supplementary_class directory, unzip to get doc.kml and put in appropriate subdirectory.
2. Run ShapefileHelper.py to run the method ingest_kml_fixed_classes.


##Model parameterization
###Model_Evaluator.py has 2 main methods: evaluate_model() and evaluate_bands() 
###Run grid searches iteratively saving parameterizations to a SQL-based database
1.evaluate_model() can be run to save the best parameterizations from a grid search for sets of concessions to a sql-lite-database
2. Find param_grid in train_model() and adjust for any parameters you would like to try.
3. Adjust concessions which should or should not be included in the trials by changing the sites variable at the top.


###Run Models with different sets of bands to evaluate input data
1. evaluate_bands() cn be run to ...


##Classification
###Run Large Area classifier with configuration and data from above modules.
1.


##Post-processing
###Cleanup and write final full island maps
1.



