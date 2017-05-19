var baseLayer = new ol.layer.Group({
    'title': '',
    layers: [
new ol.layer.Tile({
    'title': 'OSM B&W',
    'type': 'base',
    source: new ol.source.XYZ({
        url: 'http://{a-c}.www.toolserver.org/tiles/bw-mapnik/{z}/{x}/{y}.png',
        attributions: [new ol.Attribution({html: '&copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>'})]
    })
})
]
});
var format_results0 = new ol.format.GeoJSON();
var features_results0 = format_results0.readFeatures(json_results0, 
            {dataProjection: 'EPSG:4326', featureProjection: 'EPSG:3857'});
var jsonSource_results0 = new ol.source.Vector({
    attributions: [new ol.Attribution({html: '<a href=""></a>'})],
});
jsonSource_results0.addFeatures(features_results0);var lyr_results0 = new ol.layer.Vector({
                source:jsonSource_results0, 
                style: style_results0,
                title: "results"
            });var format_beijing_23K1 = new ol.format.GeoJSON();
var features_beijing_23K1 = format_beijing_23K1.readFeatures(json_beijing_23K1, 
            {dataProjection: 'EPSG:4326', featureProjection: 'EPSG:3857'});
var jsonSource_beijing_23K1 = new ol.source.Vector({
    attributions: [new ol.Attribution({html: '<a href=""></a>'})],
});
jsonSource_beijing_23K1.addFeatures(features_beijing_23K1);var lyr_beijing_23K1 = new ol.layer.Vector({
                source:jsonSource_beijing_23K1, 
                style: style_beijing_23K1,
                title: "beijing_23K"
            });

lyr_results0.setVisible(true);lyr_beijing_23K1.setVisible(true);
var layersList = [baseLayer,lyr_results0,lyr_beijing_23K1];
lyr_results0.set('fieldAliases', {'field_1': 'field_1', 'field_2': 'field_2', 'field_3': 'field_3', });
lyr_beijing_23K1.set('fieldAliases', {'field_1': 'field_1', 'field_2': 'field_2', 'field_3': 'field_3', 'field_4': 'field_4', 'field_5': 'field_5', });
lyr_results0.set('fieldImages', {'field_1': 'TextEdit', 'field_2': 'TextEdit', 'field_3': 'TextEdit', });
lyr_beijing_23K1.set('fieldImages', {'field_1': 'TextEdit', 'field_2': 'TextEdit', 'field_3': 'TextEdit', 'field_4': 'TextEdit', 'field_5': 'TextEdit', });
lyr_results0.set('fieldLabels', {'field_1': 'no label', 'field_2': 'no label', 'field_3': 'no label', });
lyr_beijing_23K1.set('fieldLabels', {'field_1': 'no label', 'field_2': 'no label', 'field_3': 'no label', 'field_4': 'no label', 'field_5': 'no label', });
lyr_beijing_23K1.on('precompose', function(evt) {
    evt.context.globalCompositeOperation = 'normal';
});