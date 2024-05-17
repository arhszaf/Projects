package src;

import java.io.*;
import java.net.*;
import java.util.*;
import org.xml.sax.InputSource;

/* βάλαμε μέσα στην while του toClient το fromWorker = new... και μας εβγαζε corrupt.
Οταν κλειναμε το fromWorker μεσα στο while μας εβγαζε socket closed */

public class Master implements Runnable{

    public static void main(String args[]) {
        new Master().openServer();
    }

    ServerSocket ss1;
    ServerSocket ss2;

    Socket socketClient;
    Socket socketWorker;
    static int numberWorkers = 0;
    static List<String> workerIpList = new ArrayList<>();
    static List<Socket> workerSocket = new ArrayList<>();
    static List<List<List<String>>> chunkList;

    static Map<String, Double> intermediateResults = new HashMap<>();
    static int chunkCnt = 0;
    void openServer() {
        try {
            // read from a config file the number of workers needed to start the process
            Scanner scanner = new Scanner(new File("config.txt"));
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if (line.startsWith("number of workers")) {
                    numberWorkers = Integer.parseInt(line.substring(line.indexOf('=') + 1).trim());
                    break;
                }
            }
            scanner.close();

            //server socket for worker
            ss1 = new ServerSocket(3999);
            //server socket for client
            ss2 = new ServerSocket(6999);


            while (true) {
                System.out.println("Waiting for connection...");

                //accepts worker
                socketWorker = ss1.accept();
                String workerIp = socketWorker.getInetAddress().getHostAddress();
                System.out.println("Worker connected with IP: " + workerIp);
                workerIpList.add(workerIp);
                workerSocket.add(socketWorker);

                //starts thread to move data from worker->master->client
                toClient st2 = new toClient(socketWorker, workerIp, socketClient);
                Thread t2 = new Thread(st2);
                t2.start();

                //accepts client
                socketClient = ss2.accept();
                if(socketClient != null) {
                    System.out.println("Client connected: " + socketClient);
                    //starts thread to move data from client->master->worker
                    FromClient st1 = new FromClient(socketClient, socketWorker);
                    Thread t1 = new Thread(st1);
                    t1.start();
                }

            }
        } catch (IOException ioException) {
            ioException.printStackTrace();
        } finally {
            try{
                ss1.close();
                ss2.close();
            } catch (IOException ioException) {
                ioException.printStackTrace();
            }
        }
    }

    @Override
    public void run() {

    }

    public static class FromClient implements Runnable,Serializable {

       InputStream in;
        Socket socketClient;
        Socket socketWorker;
        int currentIndex = 0;


        public FromClient(Socket socketClient,  Socket socketWorker) {
            this.socketClient = socketClient;
            this.socketWorker = socketWorker;
        }

        @Override
        public void run() {
            try {
                // creates a new file to store the gpx from the client
                File file = new File("file.gpx");
                try {
                    if (file.createNewFile()) {
                        System.out.println("File created successfully.");
                    } else {
                        System.out.println("File already exists.");
                    }
                } catch (IOException e) {
                    System.err.println("Error creating file: " + e.getMessage());
                }

                // receives the gpx from the client and writes it to the created file
                in = socketClient.getInputStream();
                InputStreamReader reader = new InputStreamReader(in, "UTF-8");
                DataInputStream dataInputStream = new DataInputStream(in);
                // Receive the file content from the client
                String receivedFileContent = dataInputStream.readUTF();
                 // Write the received content to a file
                File outputFile = new File("file.gpx");
                writeStringToFile(receivedFileContent, outputFile);
                //dataInputStream.close();



                // parses the file
                GPXParser GPXParsing = new GPXParser("file.gpx");
                List<List<String>> wptList = GPXParsing.getChunkList();

                // creates a List<List<List<String>>>
                chunkList = new ArrayList<>();
                List<List<String>> temp = new ArrayList<>();

                // creates chunks of List<List<String>>
                for(int i = 0; i < wptList.size(); i ++) {
                    temp.add(wptList.get(i));
                    if((i+1) % 12 == 0) {
                        chunkList.add(temp);
                        temp = new ArrayList<>();
                    }
                }


                if(!temp.isEmpty()) {
                    chunkList.add(temp);
                }

                // for every chunk in the chunkList call round-robin
                for(List<List<String>> chunk : chunkList) {
                    roundRobinSend(chunk, workerSocket);
                }


            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        private void writeStringToFile(String content, File file) throws IOException {
            BufferedWriter writer = new BufferedWriter(new FileWriter(file));
            writer.write(content);
            writer.close();
        }


        private void roundRobinSend(List<List<String>> chunk, List<Socket> socketWorker) {
            int numWorker = workerIpList.size();
            ObjectOutputStream pp;

            // worketSocket swaps between the sockets of workers
            Socket workerSocket = socketWorker.get(currentIndex);
            currentIndex = (currentIndex + 1) % numWorker;

            // waits for needed workers to connect
            while(numWorker < numberWorkers) {

            }
            try {
                pp = new ObjectOutputStream(workerSocket.getOutputStream());
                // increments every time a chunk is sent out
                chunkCnt++;
                pp.writeObject(chunk);
                pp.flush();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

        }


    }

    public static class toClient implements Runnable,Serializable{
        ObjectInputStream fromWorker;
        ObjectOutputStream out;
        Socket socketWorker;
        Socket socketClient;
        String workerIp;
        public toClient(Socket socketWorker, String workerIp, Socket socketClient) {
            this.socketWorker = socketWorker;
            this.socketClient = socketClient;
            this.workerIp = workerIp;
        }



        @Override
        public void run() {

            out = null;

            try {

                //gets intermediate results from worker
                //out = new ObjectOutputStream(socketClient.getOutputStream());
                fromWorker = new ObjectInputStream(socketWorker.getInputStream());


                while (true) {

                    //read data from worker
                    Map<String, Double> dataFromWorker = (Map<String, Double>) fromWorker.readObject();
                    System.out.println("Recieved from " + workerIp + " this map: " + dataFromWorker);

                    // decrements after we receive a processed chunk from the worker
                    chunkCnt--;

                    // when all chunks have been received get out of the loop
                    if(chunkCnt == 0) {
                        break;
                    }

                    //adds data into hashmap intermediate results
                    for (Map.Entry<String, Double> entry : dataFromWorker.entrySet()) {
                        String key = entry.getKey();
                        Double value = entry.getValue();
                        if (intermediateResults.containsKey(key)) {
                            // if the key already exists in the intermediate results, add the new value to the existing value
                            Double oldValue = intermediateResults.get(key);
                            // converting time from seconds to minutes
                            if(key.equals("Total time")) {
                                value =  value / 60;
                            }
                            intermediateResults.put(key, (oldValue + value));
                        } else {

                            if(key.equals("Total time")) {
                                value = value / 60;
                            }
                            // if the key does not exist in the intermediate results, add the new key-value pair
                            intermediateResults.put(key, value);
                        }
                    }
                    System.out.println("Inter Results Map: " + intermediateResults + " as far.");

                }

                if(socketClient != null) {
                    out = new ObjectOutputStream(socketClient.getOutputStream());
                }

                // sends the final results to client
                if(out !=null) {
                    // calculates the average speed
                    intermediateResults.put("Avg speed", intermediateResults.get("Avg speed") / chunkList.size());
                    out.writeObject(intermediateResults);
                    System.out.println("Sent this map: " + intermediateResults);
                    out.flush();
                }

            }catch(IOException | ClassNotFoundException e){
                throw new RuntimeException(e);
            } finally{
                try {
                    fromWorker.close();
                    if(out != null) {
                        out.close();
                    }

                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }



}
