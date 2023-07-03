

# Set up ------------------------------------------------------------------

# Libraries
library(tidyverse)
library(sf)
library(raster)
library(ncdf4)
library(mapview)
library(jtools)  

map <- read_sf('methane_processing/data/cb_2018_us_county_500k/cb_2018_us_county_500k.shp')

# Sources
# Map: https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
# ISAI L2:   https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-methane?tab=form
  # documentation:http://wdc.dlr.de/C3S_312b_Lot2/Documentation/GHG/C3S2_312a_Lot2_PUGS_GHG_E_latest.pdf
  # more supporting documentationL https://amt.copernicus.org/articles/10/4623/2017/amt-10-4623-2017.pdf


# IASI L2 -----------------------------------------------------------------

# Request data -> jupyter notebook

# We have one file for each day that we requested 

methane <- 
  data.frame(
      latitude = double(),
      longitude = double(), 
      ch4 = double(), 
      time = POSIXct(), 
    stringsAsFactors=FALSE)

files <- list.files('methane_processing/data/methane2019/')

for (i in files) {

  # Open connection to file 
  nc <- nc_open(paste0('methane_processing/data/methane2019/', i))
  
  # Extract variable 
  time <- ncvar_get(nc, varid='time') # seconds since 1970-1-1 0:0:0
  ch4 <- ncvar_get(nc, varid='ch4') # unit: 1e-9
  lat <- ncvar_get(nc, varid='latitude') # degrees north -90f, 90f, geospatial vertical_min: 0.05, max: 1013,25
  lon <- ncvar_get(nc, varid='longitude') # degrees east -180f, 180f
  
  # Turn into a data frame 
  df <- data.frame(latitude = lat, longitude = lon, ch4 = ch4, time = as.POSIXct(time, origin = '1970-01-01'))
  
  # Append files 
  methane <- rbind(methane, df)
  print(i)
  
  nc_close(nc)

}

methane_county <- 
  methane %>% 
  st_as_sf(dim='XY', coords=c('longitude', 'latitude'), crs=crs(map)) %>%
  st_join(map, left=F) %>% 
  mutate(longitude = unlist(map(geometry,1)),
         latitude = unlist(map(geometry,2))) %>% 
  as.data.frame() %>% 
  dplyr::select(-geometry) 

write_csv(methane_county, 'methane_processing/final_csvs/methane_daily2019_county.csv')
write_csv(methane, 'methane_processing/final_csvs/methane_daily2019.csv')

# There are not always the same number of observations made in a day. Very different longitude values

methane %>%
  filter(
    latitude > 19 & latitude < 54 &
    longitude > -135 & longitude < -50 ) %>% 
  mutate(
    date = as.Date(time) ) %>% 
  mapview(xcol = "longitude", ycol = "latitude", zcol='ch4', crs = 4269, grid = FALSE)


# L3 ----------------------------------------------------------------------

# Request data -> jupyter notebook

# We have an array with monthly data for longitude and latitude 

nc <- nc_open('methane_processing/data/200301_202112-C3S-L3_GHG-GHG_PRODUCTS-MERGED-MERGED-OBS4MIPS-MERGED-v4.4.nc')


time <- ncvar_get(nc, varid='time') # days since 1970-1-1 0:0:0
ch4 <- ncvar_get(nc, varid='xch4') # unit: 1e-9
lat <- ncvar_get(nc, varid='lat') # 
lon <- ncvar_get(nc, varid='lon') # 

dim(ch4) # 72 x 36 x 228

# Note that these have different lengths and we therefore cannot simply create a data frame with them as we did above
length(lat) # 36
length(lon) # 72
length(time) # 228

dimnames(ch4) <- list(lon, lat, time)
df <- as.data.frame.table(ch4) 
colnames(df) <- c('longitude', 'latitude', 'time', 'ch4')

df$time = as.numeric(as.character(df$time))
df$date = as.Date(df$time, origin = '1990-01-01')

df <- df %>% 
  mutate(
    latitude = as.numeric(as.character(latitude)), 
    longitude = as.numeric(as.character(longitude)) )

df_county <- 
  df %>% 
  st_as_sf(dim='XY', coords=c('longitude', 'latitude'), crs=crs(map)) %>%
  st_join(map, left=F) %>% 
  mutate(longitude = unlist(map(df_match$geometry,1)),
         latitude = unlist(map(df_match$geometry,2))) %>% 
  as.data.frame() %>% 
  dplyr::select(-geometry) 

# Check geographic density of these observations 
df %>%
  filter(
    latitude > 19 & latitude < 54 &
      longitude > -135 & longitude < -50 ) %>% 
  mapview(xcol = "longitude", ycol = "latitude", zcol='ch4', crs = 4269, grid = FALSE)

# Plot methane over time 
df %>% 
  filter(
    latitude > 19 & latitude < 54 &
      longitude > -135 & longitude < -50 ) %>% 
  na.omit() %>% 
  ggplot(aes(x=date, y = ch4)) + 
  geom_point() + 
  geom_smooth(method = 'loess') + 
  theme_nice()

write_csv(df, 'methane_processing/final_csvs/methane_monthly.csv')
write_csv(df_county, 'methane_processing/final_csvs/methane_monthly_county.csv')

