# GPX Data Processing System Using MapReduce Framework
### Overview
This project implements a GPX data processing system using a MapReduce framework. It reads GPX files from a client, processes them through multiple worker nodes, and returns the calculated statistics back to the client. The system is implemented using Java sockets for communication between the client, master, and worker nodes.

### Components
The system consists of four main components:

**Client**: Sends GPX file data to the master server and receives the processed results.
**Master**: Manages the connection between clients and workers, distributes the workload, and aggregates the results from workers.
**Worker**: Processes chunks of GPX data to compute various statistics like total distance, average speed, total elevation, and total time.
**GPXParser**: Parses the GPX file to extract waypoint information.
