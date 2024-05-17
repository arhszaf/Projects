package src;

import org.w3c.dom.*;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class GPXParser {


    List<List<String>> chunkList;
    private int numWpt;

    private void GPXParsing(String filename) {

        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        try {
            //Makes document builder
            DocumentBuilder builder = factory.newDocumentBuilder();

            //Gets document
            Document document = builder.parse(new File(filename));

            //Normalizes xml structure
            document.getDocumentElement().normalize();

            //Get all the element by the tag name
            NodeList wptList = document.getElementsByTagName("wpt");

            for(int i = 0; i < wptList.getLength(); i++) {
                //gets the gpx chunks
                Node wpt = wptList.item(i);

                List<String> innerList = new ArrayList<>();

                // gets attributes and elements wpt
                if (wpt.getNodeType() == Node.ELEMENT_NODE) {
                    Element wptEle = (Element) wpt;
                    NamedNodeMap attributes = wptEle.getAttributes();
                    Node latNode = attributes.getNamedItem("lat");
                    String attributeLat = latNode.getNodeName();

                    Node lonNode = attributes.getNamedItem("lon");
                    String attributeLon = lonNode.getNodeName();

                    String lat = wptEle.getAttribute("lat");
                    String lon = wptEle.getAttribute("lon");
                    String ele = wptEle.getElementsByTagName("ele").item(0).getTextContent();
                    String time = wptEle.getElementsByTagName("time").item(0).getTextContent();

                    innerList.add(attributeLat);
                    innerList.add(lat);
                    chunkList.add(innerList);

                    innerList = new ArrayList<>();
                    innerList.add(attributeLon);
                    innerList.add(lon);
                    chunkList.add(innerList);

                    // access the ele and time nodes
                    NodeList wptData = wpt.getChildNodes();
                    for (int j=0; j < wptData.getLength(); j++) {
                        Node data = wptData.item(j);
                        innerList = new ArrayList<>();
                        if(data.getNodeType() == Node.ELEMENT_NODE) {
                            Element dataEle = (Element) data;
                            if(data.getNodeName().equals("ele")) {

                                innerList.add(dataEle.getTagName());
                                innerList.add(ele);
                            } else {

                                innerList.add(dataEle.getTagName());
                                innerList.add(time);
                            }

                            chunkList.add(innerList);
                        }
                    }
                }
            }
            numWpt = wptList.getLength();

        } catch (ParserConfigurationException | SAXException | IOException e) {
            e.printStackTrace();
        }
    }

    public  GPXParser(String filename) {
        this.chunkList = new ArrayList<>();
        GPXParsing(filename);
    }


    public List<List<String>> getChunkList() {
        return this.chunkList;
    }


}
