package Misc;

import com.github.filosganga.geogson.gson.GeometryAdapterFactory;
import com.github.filosganga.geogson.model.*;
import com.google.common.collect.ImmutableMap;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonPrimitive;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class GeoGSON {
    private BufferedWriter bw;
    private FileWriter fw;
    private Gson gson;
    private StringBuilder geojson;
    private List<Feature> features;

    public GeoGSON(String crs){
        this.bw = null;
        this.fw = null;
        this.gson = new GsonBuilder()
                .registerTypeAdapterFactory(new GeometryAdapterFactory())
                .setPrettyPrinting()
                .create();
        this.geojson = new StringBuilder();
        this.geojson.append("{\"type\": \"FeatureCollection\",\n")
                .append("\"crs\": { \"type\": \"name\", \"properties\": { \"name\": \"urn:ogc:def:crs:EPSG::")
                .append(crs)
                .append("\" } },\n");
        features = new ArrayList<>();
    }

    public void makeMBRs(Double lon1, Double lat1, Double lon2, Double lat2, Integer id, Integer popup){
        LinearRing l = LinearRing.of(
                Point.from(lon1, lat1),
                Point.from(lon1, lat2),
                Point.from(lon2, lat2),
                Point.from(lon2, lat1),
                Point.from(lon1, lat1));
        Polygon p = Polygon.of(l);
        JsonElement jsonElement = new JsonPrimitive(popup);
        Feature f = Feature.of(p).withProperties(ImmutableMap.of("popup", jsonElement)).withId(id.toString());
        features.add(f);
    }
    
    public void makePoints(Double lon, Double lat){
        features.add(Feature.of(Point.from(lon, lat)));
    }

    public void saveGeoJSON(String filename){
        FeatureCollection collection = new FeatureCollection(features);
        String features = gson.toJson(collection).substring(1);
        this.geojson.append(features);

        try {
            fw = new FileWriter(filename);
            bw = new BufferedWriter(fw);
            bw.write(this.geojson.toString());
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (bw != null)
                try {
                    bw.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            if (fw != null)
                try {
                    fw.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
        }
    }

    public static void main(String[] args) {
        Double lon1 = 10.0;
        Double lat1 = 10.0;
        Double lon2 = 20.0;
        Double lat2 = 20.0;
        Integer id = 1;

        GeoGSON gson1 = new GeoGSON("4799");
        gson1.makeMBRs(lon1, lat1, lon2, lat2, id, 5);
        gson1.saveGeoJSON("/tmp/MBRs.json");

        GeoGSON gson2 = new GeoGSON("3068");
        gson2.makePoints(lon1, lat1);
        gson2.saveGeoJSON("/tmp/Points.json");
    }
}

