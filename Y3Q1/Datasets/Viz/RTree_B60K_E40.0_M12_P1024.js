function PointsStyle(feature) {
	var style = new ol.style.Style({
		image: new ol.style.Circle({
			radius: 2,
			stroke: new ol.style.Stroke({
				color: 'black',
				width: 1
			}),
			fill: new ol.style.Fill({
				color: 'red'
			})
		})
	})
	return [style];
}

function MBRsStyle(feature) {
	var style = new ol.style.Style({
		stroke: new ol.style.Stroke({
			color: 'orange',
			lineDash: [5],
			width: 2
		}),
		fill: new ol.style.Fill({
			color: 'rgba(0, 0, 255, 0.05)'
		})
	})
	return [style];
}

var MBRsSource =  new ol.source.Vector({
	format: new ol.format.GeoJSON(),
	projection : 'urn:ogc:def:crs:EPSG::3068',
	url: 'RTree_B60K_E40.0_M12_P1024.geojson'
})

var MBRsLayer = new ol.layer.Vector({
	title: 'MBRs',
	source: MBRsSource,
	style: MBRsStyle
});

var PointsSource =  new ol.source.Vector({
	format: new ol.format.GeoJSON(),
	projection : 'urn:ogc:def:crs:EPSG::3068',
	url: 'Points_B60K_E40.0_M12_P1024.geojson'
})

var PointsLayer = new ol.layer.Vector({
	title: 'Candidate disks',
	source: PointsSource,
	style: PointsStyle
});

var cx = 25521.486 
var cy = 20836.726
var extend = 500000
proj4.defs("EPSG:3068","+proj=cass +lat_0=52.41864827777778 +lon_0=13.62720366666667 +x_0=40000 +y_0=10000 +ellps=bessel +datum=potsdam +units=m +no_defs");
var proj = ol.proj.get('EPSG:3068');
proj.setExtent([cx - extend, cy - extend, cx + extend, cy + extend]);    

var map = new ol.Map({
	layers: [
		new ol.layer.Tile({
			title: 'OSM',
			type: 'base',
			visible: false,
			source: new ol.source.OSM()
		}),
		new ol.layer.Tile({
			title: 'Bing',
			type: 'base',
			visible: true,
			source: new ol.source.BingMaps({
				key: 'AnBTYXxmiTX1rqG4gmggxSEzegeK5P1KLG6oF7EJhUuKgI-cS8UefgI5mUTvVHTo',
				imagerySet: 'AerialWithLabels',
				maxZoom: 18
			})
		}),
		new ol.layer.Tile({
			title: 'Blank',
			type: 'base',
			visible: false,
			source: null
		}),
		PointsLayer,
		MBRsLayer
	],
	target: 'map',
	controls: ol.control.defaults({
		attributionOptions: ({
			collapsible: true
		})
	}),
	view: new ol.View({
		projection: 'EPSG:3068',
		center: [cx, cy],
		zoom: 7
	})
});
var layerSwitcher = new ol.control.LayerSwitcher({
	tipLabel: 'Legend' // Optional label for button
});
map.addControl(layerSwitcher);
