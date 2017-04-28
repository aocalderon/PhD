var image = new ol.style.Circle({
	radius: 2,
	fill: new ol.style.Fill({
		color: 'rgba(0, 0, 255, 0.5)'
	})
});

var styles = {
	'Point': new ol.style.Style({
		image: image
	}),
	'Polygon': new ol.style.Style({
		stroke: new ol.style.Stroke({
			color: 'blue',
			lineDash: [4],
			width: 3
		}),
		fill: new ol.style.Fill({
			color: 'rgba(0, 0, 255, 0.1)'
		})
	}),
	'Circle': new ol.style.Style({
		stroke: new ol.style.Stroke({
			color: 'red',
			lineDash: [4],
			width: 1
		}),
		fill: new ol.style.Fill({
			color: 'rgba(255,0,0,0.1)'
		})
	})
};

var styleFunction = function(feature) {
	return styles[feature.getGeometry().getType()];
};

var geojsonObject = {
	'type': 'FeatureCollection',
	'crs': {
		'type': 'name',
		'properties': {
			'name': 'EPSG:3857'
		}
	},
	'features': [{
		'type': 'Feature',
		'geometry': {
			'type': 'Point',
			'coordinates': [0, 0]
		}
	}]
};

var vectorSource =  new ol.source.Vector({
	format: new ol.format.GeoJSON(),
	projection : 'EPSG:3857',
	url: 'http://localhost:8000/demo.geojson'
})

vectorSource.addFeature(new ol.Feature(new ol.geom.Circle([5e6, 7e6], 1e6)));

var vectorLayer = new ol.layer.Vector({
	source: vectorSource,
	style: styleFunction
});

var map = new ol.Map({
	layers: [
		new ol.layer.Tile({
		source: new ol.source.OSM()
		}),
		vectorLayer
	],
	target: 'map',
	controls: ol.control.defaults({
		attributionOptions: ({
			collapsible: true
		})
	}),
	view: new ol.View({
		center: [0, 0],
		zoom: 3
	})
});
