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
var format_Points0 = new ol.format.GeoJSON();
var features_Points0 = format_Points0.readFeatures(geojson_Points0, 
            {dataProjection: 'EPSG:4326', featureProjection: 'EPSG:3857'});
var jsonSource_Points0 = new ol.source.Vector({
    attributions: [new ol.Attribution({html: '<a href=""></a>'})],
});
jsonSource_Points0.addFeatures(features_Points0);var lyr_Points0 = new ol.layer.Vector({
                source:jsonSource_Points0, 
                style: style_Points0,
                title: "Points"
            });var format_Buffers1 = new ol.format.GeoJSON();
var features_Buffers1 = format_Buffers1.readFeatures(geojson_Buffers1, 
            {dataProjection: 'EPSG:4326', featureProjection: 'EPSG:3857'});
var jsonSource_Buffers1 = new ol.source.Vector({
    attributions: [new ol.Attribution({html: '<a href=""></a>'})],
});
jsonSource_Buffers1.addFeatures(features_Buffers1);var lyr_Buffers1 = new ol.layer.Vector({
                source:jsonSource_Buffers1, 
                style: style_Buffers1,
                title: "Buffers"
            });var format_MBRs2 = new ol.format.GeoJSON();
var features_MBRs2 = format_MBRs2.readFeatures(geojson_MBRs2, 
            {dataProjection: 'EPSG:4326', featureProjection: 'EPSG:3857'});
var jsonSource_MBRs2 = new ol.source.Vector({
    attributions: [new ol.Attribution({html: '<a href=""></a>'})],
});
jsonSource_MBRs2.addFeatures(features_MBRs2);var lyr_MBRs2 = new ol.layer.Vector({
                source:jsonSource_MBRs2, 
                style: style_MBRs2,
                title: "MBRs"
            });

lyr_Points0.setVisible(true);lyr_Buffers1.setVisible(true);lyr_MBRs2.setVisible(true);
var layersList = [baseLayer,lyr_Points0,lyr_Buffers1,lyr_MBRs2];
lyr_Points0.set('fieldAliases', {'field_1': 'field_1', 'field_2': 'field_2', 'field_3': 'field_3', });
lyr_Buffers1.set('fieldAliases', {'popup': 'popup', });
lyr_MBRs2.set('fieldAliases', {'popup': 'popup', });
lyr_Points0.set('fieldImages', {'field_1': 'TextEdit', 'field_2': 'TextEdit', 'field_3': 'TextEdit', });
lyr_Buffers1.set('fieldImages', {'popup': 'TextEdit', });
lyr_MBRs2.set('fieldImages', {'popup': 'TextEdit', });
lyr_Points0.set('fieldLabels', {'field_1': 'no label', 'field_2': 'no label', 'field_3': 'no label', });
lyr_Buffers1.set('fieldLabels', {'popup': 'no label', });
lyr_MBRs2.set('fieldLabels', {'popup': 'no label', });
lyr_MBRs2.on('precompose', function(evt) {
    evt.context.globalCompositeOperation = 'normal';
});