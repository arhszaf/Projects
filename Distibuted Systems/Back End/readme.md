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
The GPX file should contain waypoints (<wpt>) with latitude (lat), longitude (lon), elevation (<ele>), and time (<time>) elements. Here's an example:
```
<gpx>
  <wpt lat="47.644548" lon="-122.326897">
    <ele>4.46</ele>
    <time>2020-01-01T00:00:00Z</time>
  </wpt>
</gpx>
```
