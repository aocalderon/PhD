library(shiny)
library(leaflet)

ui <- fluidPage(
  leafletOutput("mymap"),
  actionButton("reset_button", "Reset view")
  
)

server <- function(input, output, session) {
  initial_lat = -23.079
  initial_lng = 178.15
  initial_zoom = 4
  
  output$mymap <- renderLeaflet({ leaflet(quakes) %>% 
      setView(lat = initial_lat, lng = initial_lng, zoom = initial_zoom) %>%
      addTiles(group = "OSM (default)") %>%
      addProviderTiles("Stamen.Toner", group = "Toner") %>%
      addProviderTiles("Stamen.TonerLite", group = "Toner Lite")})                                 
  
  
  observe({
    input$reset_button
    leafletProxy("mymap") %>% setView(lat = initial_lat, lng = initial_lng, zoom = initial_zoom)
  })
}

shinyApp(ui, server)
