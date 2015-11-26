require(sqldf)
require(data.table)
## stations <- read.csv('WeatherStationLocations.csv', header = T, sep = ",", quote = '"')
data <- read.csv('data.csv', header = F)

stations_us <- stations[stations$STATE != '' & stations$CTRY == 'US', c('USAF', 'STATE', 'LAT', 'LON')]
results <- sqldf("SELECT * FROM data JOIN stations_us ON data.V1 = stations_us.USAF")
results$V2 <- strptime(results$V2, format = "%Y%m%d")
results$month <- format(results$V2,'%m')
results$year <- format(results$V2,'%Y')
head(results)
table(as.character(results$STATE),format(results$V2,'%m'))
dt <- data.table(results)
as.data.frame(dt[, list(max(V3), min(V3), max(V3) - min(V3)), by = list(STATE, month)])
