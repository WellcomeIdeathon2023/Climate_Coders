####Simulated Data####

#Welcome the script which will produce a simulated dataset of methane labs 
# and hospitals located across the USA


#Set seed for reproducibility
set.seed(4321)

#This script will produce observations of 300 fossil fuel extraction and 
#processing plants


#There will be 7 variables: longitude, latitude, company_name, fossil_fuel_type, 
#extraction_or_processing, number_of_employees, previous_leaks_n

# longitude - integer variable. Represents longitude of site. Generated from
#   random number between -124 and -66
# latitude - integer variable. Represents longitude of site. Generated from
#   random number between 27 and 48
# company_name - categorical string variable. Represents name of company running
#   site. Generated from random selection of 15 fictional company names
# fossil_fuel_type - categorical factor variable. Represents type of fossil fuel
#   used at site: either oil, coal or natural gas - randomly sampled.
# extraction_or_processing - categorical factor variable. Represents process at
#   site: either processing or extracting fuel - randomly sampled.
# number_of_employees - integer variable. Represents number of employees at
#   site. Generated by random number between 200 and 10,000
# number_of_beds - integer variable. Represents number of hospital beds at
#   site. Generated by random number between 50 and 2,000
# previous_leaks_n - integer variable. Represents number of previous methane 
#   leaks at site generated by sampling random numbers from 0-20`
# plant_or_hospital - categorical factor variable. Represents type of site:
#   either Energy plant or hospital - 300 each.

####Processing plants####

x <- -200.00:-66.00
longitude = sample(x, 300, replace=T)

x <- 27:80
latitude = sample(x, 300, replace=T)

x <- c("LTF", "Xeros", "Spyte", "Jeans and co", "Woodward", "Reves ltd", "Cruxx",
       "BUB Energy", "Tinted", "Leex", "Bards and Tennets", "Seets Independent",
       "Devo Minds", "Unique Oil", "COYG")
company_name = sample(x, 300, replace=T)

x <- c("Oil", "Coal", "Natural gas")
fossil_fuel_type = sample(x, 300, replace=T)

x <- c("Extraction", "Processing")
extraction_or_processing = sample(x, 300, replace=T)

x <- 200:10000
number_of_employees = sample(x, 300, replace=T)

x <- c("NA")
number_of_beds = sample(x, 300, replace=T)

x <- 0:20
previous_leaks_n = sample(x, 300, replace=T)

x <- c("Energy plant")
plant_or_hospital = sample(x, 300, replace=T)

# create data.frame
df = data.frame(
  company_name, longitude, latitude, fossil_fuel_type, extraction_or_processing,
  number_of_employees,number_of_beds, previous_leaks_n, plant_or_hospital)



#Example of mapping data

#library(mapview)

mapview(df_filtered, xcol = "longitude", ycol = "latitude", crs = 4269, grid = FALSE)


####Hospitals####

x <- -200.00:-66.00
longitude = sample(x, 300, replace=T)

x <- 27:85
latitude = sample(x, 300, replace=T)

x <- c("Hospice Caring Ltd", "Healthy co", "CaroCare", "Lo-Care", "Well-care One", 
       "Capital Health", "Charity Trust","Well Foundation", "Med-linus", 
       "North West Health", "Nuffsons Health", "STV Health",
       "Kind", "Trusted Lab co", "Nations Health")

company_name = sample(x, 300, replace=T)

x <- c("NA")
fossil_fuel_type = sample(x, 300, replace=T)

x <- c("NA")
extraction_or_processing = sample(x, 300, replace=T)

x <- 200:10000
number_of_employees = sample(x, 300, replace=T)

x <- 50:2000
number_of_beds = sample(x, 300, replace=T)

x <- c("NA")
previous_leaks_n = sample(x, 300, replace=T)


x <- c("Hospital")
plant_or_hospital = sample(x, 300, replace=T)

# create data.frame
df2 = data.frame(
  company_name, longitude, latitude, fossil_fuel_type, extraction_or_processing,
  number_of_employees,number_of_beds, previous_leaks_n, plant_or_hospital)


df <- rbind(df, df2)

#write.csv(df, 'C:/Users/bengo/OneDrive/Documents/GitHub/Climate_Coders/data/simulated_data.csv')


# Install the rnaturalearth package if you haven't already
#install.packages("rnaturalearth")

# Load the package
library(rnaturalearth)
library(sf)

# Download the shapefile for the USA
usa <- ne_download(category = "countries", type = "united states")

usa<- ne_countries(country = "United States of America", scale = "large")

# Read the shapefile into R
usa_shapefile <- st_as_sf(usa)

library(sf)
df_sf <- st_as_sf(df, coords = c("longitude", "latitude"), crs = st_crs(usa_shapefile))

# Perform spatial intersection to keep only the rows within the USA shapefile
df_filtered <- st_intersection(df_sf, usa_shapefile)



df_filtered <- as.data.frame(df_filtered[c(1:7)])%>% dplyr::select(-geometry)

df <- merge(df_filtered, df[c("company_name", "number_of_employees", "longitude", "latitude" )], 
            by=c("company_name", "number_of_employees"), all.x=T)

write.csv(df, 'C:/Users/bengo/OneDrive/Documents/GitHub/Climate_Coders/data/simulated_data.csv')









