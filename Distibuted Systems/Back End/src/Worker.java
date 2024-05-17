package src;

import java.time.Duration;
import java.util.*;
import java.net.*;
import java.io.*;
import java.time.Instant;
import java.io.Serializable;
import static java.lang.Math.*;

public class Worker implements Serializable{

    static List<List<String>> list;
    static Map<String ,Double> map;
    static double distance;
    static double totalElevation;

    static Socket requestSocket;
    static Object monitor;
    static int counter=0;


    // Socket requestSocket = null;


    public Worker() {

    }



    public static void main(String[] args) throws IOException {
        ObjectOutputStream output=null;
        ObjectInputStream input=null;

        try {

            requestSocket = new Socket("192.168.56.1", 3999);
            requestSocket.setSoTimeout(10000);
            output = new ObjectOutputStream(requestSocket.getOutputStream());
            //Create HashMap to produce new <Key,Value> pair
            map = new HashMap<>();
            while (true) {
                try {
                    input = new ObjectInputStream(requestSocket.getInputStream());

                    System.out.println("Remote socket address: " + requestSocket.getRemoteSocketAddress());
                    System.out.println("Connection with thread:1 established");
                    // takes input
                    list = (List<List<String>>) input.readObject();


                    //We use monitor to wait for all threads to finish and then send them back to the Master
                    monitor = new Object();


                    System.out.println("Received list from server: " + list);
                    //create 4 threads and assign each thread to a task
                    Total_distance distance = new Total_distance();
                    Thread th1 = new Thread(distance);
                    Avg_speed speed = new Avg_speed();
                    Thread th2 = new Thread(speed);
                    Total_elevation elevation = new Total_elevation();
                    Thread th3 = new Thread(elevation);
                    Total_time time = new Total_time();
                    Thread th4 = new Thread(time);
                    //we use synchronize in monitor object to ensure that the threads are notified properly
                    synchronized (monitor) {
                        th1.start();
                        th2.start();
                        th3.start();
                        th4.start();
                        //we have a counter to ensure that all threads are finished with the calculations
                        while (counter < 4) {
                            try {
                                monitor.wait();
                            } catch (InterruptedException e) {
                                // Handle exception
                            }
                            counter++;
                        }
                    }
                    System.out.println("Sending this map: " + map);
                    output.writeObject(map);
                    System.out.println("i am here");
                    output.flush();
                    map.clear();
                    list.clear();
                    counter = 0;
                } catch (IOException e) {
                    System.out.println("Connection closed by server");
                    break;
                } catch (ClassNotFoundException e) {
                    throw new RuntimeException(e);
                }


            }

        } catch (UnknownHostException e) {
            System.err.println("You are trying to connect to unknown host!");

        } catch (IOException e) {
            e.printStackTrace();
        }finally {

            if(input!=null){
                input.close();
            }

            if(output!=null) {
                output.close();
            }
            requestSocket.close();

        }
    }





    public synchronized double calculateDistance() {
        // code that calculates distance
        int i =0;
        double latDistance=0.0;
        double lonDistance=0.0;
        double a=0.0;
        double c=0.0;
        final double earth_radius = 6371;
        distance=0.0;
        while(i< list.size()-4){
            lonDistance = Math.toRadians(abs(Double.parseDouble(list.get(i+5).get(1)) - Double.parseDouble(list.get(i+1).get(1))));
            a =abs(Math.sin(latDistance / 2) * Math.sin(latDistance / 2) + Math.cos(Math.toRadians(Double.parseDouble(list.get(i).get(1)))) * Math.cos(Math.toRadians(Double.parseDouble(list.get(i+4).get(1)))) * Math.sin(lonDistance / 2) * Math.sin(lonDistance / 2));
            c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
            distance =distance+(earth_radius * c);
            i=i+4;
        }

        return distance;

}
    public double getTotalDistance(){
        return distance;
    }

    public synchronized double totalTime(){
        Instant instant1 = null;
        Instant instant2 = null;
        instant1 = Instant.parse(list.get(3).get(1));
        instant2 = Instant.parse(list.get(list.size()-1).get(1));
        Duration duration =  Duration.between(instant1,instant2);
        return duration.getSeconds();
    }
    /*convert time to minutes and then return it*/
    public synchronized double getTime() {
        double duration = totalTime();
        return duration;


    }
    /*calculate average speed */
    public synchronized double avgSpeed(){
        double duration = totalTime();
        double elapsedSeconds=0.0;
        elapsedSeconds = duration;
        return  getTotalDistance() / elapsedSeconds* 3600.0 ; // calculate average speed;

    }
    public synchronized double totalElevation(){
        int i = 2;
        double diff=0.0;
        totalElevation=0.0;
        while(i<=list.size()-4) {

            diff = Math.max(0, Double.parseDouble(list.get(i+4).get(1)) - Double.parseDouble(list.get(i).get(1)));
            totalElevation += diff;
            i=i+4;

        }
        return totalElevation;
    }


}
//different classes to assign each thread to a task
class Avg_speed extends Worker implements Runnable{

    @Override
    public void run() {

        synchronized (map){
            System.out.println("Avg speed");
            map.put("Avg speed", avgSpeed());
            System.out.println(map);
            synchronized (monitor) {
                monitor.notify();
            }
        }
    }

}
class Total_distance extends Worker implements Runnable{
    @Override
    public void run(){
        synchronized (map){
            System.out.println("Total distance");
            map.put("Total Distance",calculateDistance());
            System.out.println(map);
            synchronized (monitor){
                monitor.notify();
            }
        }
    }
}
class Total_elevation extends Worker implements Runnable{
    @Override
    public void run(){
        synchronized (map){
            System.out.println("total elevation");
            map.put("Total elevation",totalElevation());
            synchronized (monitor){
                monitor.notify();
            }
        }
    }

}

class Total_time extends Worker implements Runnable{
    @Override
    public void run(){
        synchronized (map){
            System.out.println("total time");
            map.put("Total time",getTime());
            System.out.println(map);
            synchronized (monitor){
                monitor.notify();
            }

        }
    }
}



