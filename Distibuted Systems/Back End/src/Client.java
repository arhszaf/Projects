package src;

import java.io.*;
import java.net.Socket;
import java.util.*;
 
public class Client {
    private static ObjectOutputStream dataOutputStream = null;

    public static void main(String[] args) throws IOException {

        Socket socket = new Socket("localhost", 6999);
        try  {


            ObjectOutputStream dataOutputStream = new ObjectOutputStream(socket.getOutputStream());
            FileInputStream fileIn = new FileInputStream("route1.gpx");



            ObjectInputStream dataInputStream = new ObjectInputStream(socket.getInputStream());
            Map<String, Double> finalResults = (Map<String, Double>) dataInputStream.readObject();
            System.out.println("Your statistics: " + finalResults);

        }
        catch (Exception e) {
            e.printStackTrace();
        } finally {
            //dataInputStream.close();
            if (dataOutputStream != null) {
                dataOutputStream.close();
            }
            if(socket != null) {
                socket.close();
                System.out.println("Close socket: " + socket);
            }
            //dataOutputStream.close();
        }
    }

}
