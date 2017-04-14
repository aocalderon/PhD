var baseLayer = new ol.layer.Group({
    'title': '',
    layers: [
new ol.layer.Tile({
    'title': 'OSM',
    'type': 'base',
    source: new ol.source.OSM()
})
]
});
var format_MBRs0 = new ol.format.GeoJSON();
var features_MBRs0 = format_MBRs0.readFeatures(geojson_MBRs0, 
            {dataProjection: 'EPSG:4326', featureProjection: 'EPSG:3857'});
var jsonSource_MBRs0 = new ol.source.Vector({
    attributions: [new ol.Attribution({html: '<a href=""></a>'})],
});
jsonSource_MBRs0.addFeatures(features_MBRs0);var lyr_MBRs0 = new ol.layer.Vector({
                source:jsonSource_MBRs0, 
                style: style_MBRs0,
                title: "MBRs"
            });var format_B1K_RTree1 = new ol.format.GeoJSON();
var features_B1K_RTree1 = format_B1K_RTree1.readFeatures(geojson_B1K_RTree1, 
            {dataProjection: 'EPSG:4326', featureProjection: 'EPSG:3857'});
var jsonSource_B1K_RTree1 = new ol.source.Vector({
    attributions: [new ol.Attribution({html: '<a href=""></a>'})],
});
jsonSource_B1K_RTree1.addFeatures(features_B1K_RTree1);var lyr_B1K_RTree1 = new ol.layer.Vector({
                source:jsonSource_B1K_RTree1, 
                style: style_B1K_RTree1,
                title: "B1K_RTree"
            });

lyr_MBRs0.setVisible(true);lyr_B1K_RTree1.setVisible(true);
var layersList = [baseLayer,lyr_MBRs0,lyr_B1K_RTree1];
lyr_MBRs0.set('fieldAliases', {'field_1': 'field_1', });
lyr_B1K_RTree1.set('fieldAliases', {'field_1': 'field_1', 'field_2': 'field_2', 'field_3': 'field_3', });
lyr_MBRs0.set('fieldImages', {'field_1': 'TextEdit', });
lyr_B1K_RTree1.set('fieldImages', {'field_1': 'TextEdit', 'field_2': 'TextEdit', 'field_3': 'TextEdit', });
lyr_MBRs0.set('fieldLabels', {'field_1': 'inline label', });
lyr_B1K_RTree1.set('fieldLabels', {'field_1': 'inline label', 'field_2': 'no label', 'field_3': 'no label', });
lyr_B1K_RTree1.on('precompose', function(evt) {
    evt.context.globalCompositeOperation = 'normal';
});