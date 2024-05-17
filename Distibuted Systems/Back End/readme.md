# GPX Data Processing System Using MapReduce Framework
### Overview
This project implements a GPX data processing system using a MapReduce framework. It reads GPX files from a client, processes them through multiple worker nodes, and returns the calculated statistics back to the client. The system is implemented using Java sockets for communication between the client, master, and worker nodes.

### Components
The system consists of four main components:

1. **Client**: Sends GPX file data to the master server and receives the processed results.

2. **Master**: Manages the connection between clients and workers, distributes the workload, and aggregates the results from workers.

3. **Worker**: Processes chunks of GPX data to compute various statistics like total distance, average speed, total elevation, and total time.

4. **GPXParser**: Parses the GPX file to extract waypoint information.

### GPX File Format
The GPX file should contain waypoints (`<wpt>`) with latitude (`<lat>`), longitude (`<lon>`), elevation (`<ele>`), and time (`<time>`) elements. Here's an example:
```
<gpx>
  <wpt lat="47.644548" lon="-122.326897">
    <ele>4.46</ele>
    <time>2020-01-01T00:00:00Z</time>
  </wpt>
</gpx>
```
### Code Structure
**GPXParser.java**: Parses the GPX file and extracts waypoints into a structured list.
**Master.java**: Handles client and worker connections, distributes chunks of data to workers, and aggregates results.
**Worker.java**: Processes assigned chunks of GPX data to compute statistics.
**Client.java**: Sends GPX file data to the master server and receives the processed results.
### MapReduce Implementation
The MapReduce framework is implemented as follows:

#### Map Phase (Worker Nodes)
* **Total_distance**: Calculates the total distance from the GPX data.
* **Avg_speed**: Computes the average speed.
* **Total_elevation**: Calculates the total elevation gain.
* **Total_time**: Computes the total time.
Each worker node processes a chunk of data and computes these statistics in parallel.

#### Reduce Phase (Master Node)
* Aggregates the results from all worker nodes.
* Computes the final statistics by combining the results from each worker.

#### Usage
1. **Client**: Sends a GPX file and waits for the processed results.
2. **Master**: Coordinates the distribution of data chunks to worker nodes and aggregates the results.
3. **Worker**: Processes data chunks to calculate total distance, average speed, total elevation, and total time.
