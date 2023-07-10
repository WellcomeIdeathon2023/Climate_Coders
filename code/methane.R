

# Set up ------------------------------------------------------------------

# Libraries
library(tidyverse)
library(sf)
library(raster)
library(ncdf4)
library(mapview)
library(jtools)  
library(viridis)

map <- read_sf('../data/cb_2018_us_county_500k/cb_2018_us_county_500k.shp')

# Sources
# Map: https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html

# IASI L2 -----------------------------------------------------------------

# Request data -> jupyter notebook
# ISAI L2:   https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-methane?tab=form
# documentation:http://wdc.dlr.de/C3S_312b_Lot2/Documentation/GHG/C3S2_312a_Lot2_PUGS_GHG_E_latest.pdf
# more supporting documentationL https://amt.copernicus.org/articles/10/4623/2017/amt-10-4623-2017.pdf

# We have one file for each day that we requested in 2019 

files <- list.files('methane_processing/data/methane_allyears/')
  
filtered_files <- files[startsWith(files, "2019")]
  
methane <- 
  data.frame(
      latitude = double(),
      longitude = double(), 
      ch4 = double(), 
      time = POSIXct(), 
    stringsAsFactors=FALSE)
  
for (i in filtered_files) {
  
    # Open connection to file 
    nc <- nc_open(paste0('methane_processing/data/methane_allyears/', i))
    
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
    
    nc_close(nc)  }
  
methane <- 
  methane %>% 
  mutate(
    date = as.Date(time), 
    month = format(date, '%m'), 
    year = format(date, '%Y') )

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

# Let's combine all these files into one 


# Let's have all lon + lat for methane and include county

methane_lonlat <- rbind(
  read_csv('methane_processing/final_csvs/methane_daily2018_county.csv', show_col_types = F), 
  read_csv('methane_processing/final_csvs/methane_daily2019_county.csv', show_col_types = F), 
  read_csv('methane_processing/final_csvs/methane_daily2020_county.csv', show_col_types = F), 
  read_csv('methane_processing/final_csvs/methane_daily2021_county.csv', show_col_types = F) )

write_csv(methane_lonlat, 'methane_processing/final_csvs/methane_final_lonlat.csv')

# Let's have methane aggregated at the county level 

methane_county <- 
  methane_lonlat %>% 
  group_by(date, COUNTYNS, STATEFP) %>% 
  reframe(
    mean_ch4 = mean(ch4, na.rm=T)
  )

write_csv(methane_county, 'methane_processing/final_csvs/methane_final_county.csv')

# Let's have methane aggregated at the state level  

methane_state <- 
  methane_lonlat %>% 
  group_by(date, STATEFP) %>% 
  reframe(
    mean_ch4 = mean(ch4, na.rm=T)
  )

write_csv(methane_state, 'methane_processing/final_csvs/methane_final_state.csv')

methane_daily <- rbind(
  read_csv('methane_processing/final_csvs/methane_daily2018.csv', show_col_types = F), 
  read_csv('methane_processing/final_csvs/methane_daily2019.csv', show_col_types = F), 
  read_csv('methane_processing/final_csvs/methane_daily2020.csv', show_col_types = F), 
  read_csv('methane_processing/final_csvs/methane_daily2021.csv', show_col_types = F) )

write_csv(methane_daily, 'methane_processing/final_csvs/methane_final_beyond.csv')


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



write_csv(df, 'methane_processing/final_csvs/methane_monthly.csv')
write_csv(df_county, 'methane_processing/final_csvs/methane_monthly_county.csv')


head(df_county)


# plot methane over time --------------------------------------------------

df_monthly <- read_csv('methane_processing/final_csvs/methane_monthly_county.csv')

# Plot methane over time 
df_monthly %>% 
  #filter(STATEFP == 48) %>% 
  ggplot(aes(x=date, y = ch4)) + 
  geom_point(aes(col=ch4)) + 
  geom_smooth(method = 'loess', col='black') + 
  theme_nice() + 
  scale_color_viridis(option='magma', direction= -1) + 
  labs(y='Methane ppb', x = 'Year', title = 'Rising methane emissions in the United States (2003-2021)')



# plot methane and deaths per month and county ----------------------------


# @ Dunja, this is where your code is 

df_county <- read_csv('../final_csvs/methane_final_county.csv')

df_health <- read_csv('../data/health_dunja/month_health_df.csv')

# GEOID in map = County Code in Dunja's. 

# Step 1: Merge GEOID back into Caro's data 

library(usmap)
library('cdlTools')

df_county <- 
  df_county %>% 
  mutate(month = format(date, '%m'), 
         year = format(date, '%Y'), 
         STATE = cdlTools::fips(STATEFP, to = "Abbreviation"))

df_county <- 
  map %>% 
  st_drop_geometry() %>% 
  dplyr::select(COUNTYNS, GEOID, NAME) %>% 
  merge(df_county, 
      by = c('COUNTYNS'))

# Define a lookup table for month abbreviations and corresponding numerical values
month_lookup <- c("Jan." = "01", "Feb." = "02", "Mar." = "03", "Apr." = "04",
                  "May" = "05", "Jun." = "06", "Jul." = "07", "Aug." = "08",
                  "Sep." = "09", "Oct." = "10", "Nov." = "11", "Dec." = "12")

# Convert the month column to numerical values
df_health$month <- month_lookup[df_health$Month]

combined <- 
  df_health %>% 
  rename(GEOID = `County Code`, 
         year = `Year Code`) %>% 
  merge(df_county %>% group_by(GEOID, month, year) %>% reframe(mean_ch4 = mean(mean_ch4, na.rm=T)), 
        by = c('GEOID', 'month', 'year') )

# @Dunja, we want to include labels for the counties in the upper quadrants 
# or be able to hover over them and see their name
combined %>% 
  filter(ID == 'methane') %>% 
  ggplot(aes(x=mean_ch4, y=Deaths)) + 
  geom_point(alpha=0.25, aes(color = (mean_ch4 > 1900) & (Deaths > 50))) + 
  #geom_label(aes(label = County), vjust = 1.5, data = subset(combined, ID == 'methane' & mean_ch4 > 1900, Deaths > 50)) +
  #geom_smooth(method = 'loess') + 
  scale_color_manual(name = 'High impact areas', values = c('black', 'red')) + 
  theme_nice() +
  labs(x='Average county-level methane emissions', y='Deaths per month', title = 'Methane-related deaths in the US') 


# @Dunja, this is where your code ends

# plot map with daily emissions -------------------------------------------

df_daily = read_csv('/Users/carolinkroeger/Library/CloudStorage/OneDrive-Nexus365/Projekte/Wellcome Ideathon/methane_processing/final_csvs/methane_final_lonlat.csv')

states <- read_sf('methane_processing/data/gadm36_USA_shp/', layer='gadm36_USA_1')


states %>% 
  st_join(
    df_daily %>% 
    mutate(year = format(date, '%Y')) %>% 
    filter(year == '2019') %>% 
    sample_n(1000) %>% 
    st_as_sf(dim='XY', coords=c('longitude', 'latitude')) %>%
    st_set_crs(st_crs(states)) ) %>%
  ggplot() + 
  geom_sf(aes(size=ch4)) +
  scale_color_viridis(option='viridis') + 
  theme_nice() + 
  theme(legend.position = 'bottom') + 
  xlim(160, 60) + ylim(0, 80)
  

# county level approach

#df_daily = read_csv('/Users/carolinkroeger/Library/CloudStorage/OneDrive-Nexus365/Projekte/Wellcome Ideathon/methane_processing/final_csvs/methane_final_county.csv')

head(df_daily)
nrow(df_daily)









