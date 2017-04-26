package main.scala;

import com.github.filosganga.geogson.gson.GeometryAdapterFactory;
import com.github.filosganga.geogson.jts.JtsAdapterFactory;
import com.github.filosganga.geogson.model.*;
import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.vividsolutions.jts.geom.GeometryFactory;
import com.vividsolutions.jts.geom.PrecisionModel;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by and on 4/25/17.
 */
public class GeoGSON {
    private String filename;
    private BufferedWriter bw;
    private FileWriter fw;
    private Gson gson;
    private StringBuilder geojson;
    private String crs;
    private List<Feature> features;

    public GeoGSON(String crs){
        this.bw = null;
        this.fw = null;
        this.gson = new GsonBuilder()
                .registerTypeAdapterFactory(new GeometryAdapterFactory())
                .create();
        this.crs = crs;
        this.geojson = new StringBuilder();
        this.geojson.append("{\"type\": \"FeatureCollection\",\n")
                .append("\"crs\": { \"type\": \"name\", \"properties\": { \"name\": \"urn:ogc:def:crs:EPSG::")
                .append(crs)
                .append("\" } },\n");
        features = new ArrayList<Feature>();
    }

    public void makeMBR(Double lon1, Double lat1, Double lon2, Double lat2, String id){
        LinearRing l = LinearRing.of(
                Point.from(lon1, lat1),
                Point.from(lon1, lat2),
                Point.from(lon2, lat2),
                Point.from(lon2, lat1),
                Point.from(lon1, lat1));
        Polygon p = Polygon.of(l);
        Feature f = Feature.of(p).withId(id);
        features.add(f);
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

    public static void main(String[] args) throws IOException {
        Double lon1 = 10.0;
        Double lat1 = 10.0;
        Double lon2 = 20.0;
        Double lat2 = 20.0;
        String id = "1";

        GeoGSON gson = new GeoGSON("4799");
        gson.makeMBR(lon1, lat1, lon2, lat2, id);
        gson.saveGeoJSON("/tmp/RTree.json");
    }
}

