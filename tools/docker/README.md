## Docker Environment

### Build image

The Dockerfile details the steps for building the testing environment. To build the Dockerfile,
make sure you have correct git ssh public and private keys under `~/.ssh/` and run the following command:

```
docker build -t pivot:latest -f ./Dockerfile . --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa)" --no-cache
```

It takes about 20-30 minutes to build the whole image. 

### Start docker containers

After building the image, run the docker container by:

```
docker run -it -d pivot:latest
```

and execute several terminals based on the started container. For example, if testing with three parties,
run six terminals by:

```
docker exec -it f3dbaa7exxxx /bin/bash
```

### Run the test algorithm

Use three executed containers as SPDZ parties and the other three containers as the pivot parties. 
For the SPDZ parties, go to `/opt/pivot/third_party/Pivot-SPDZ` path and run the following
in the three containers, respectively.

```
./semi-party.x -F -N 3 -I -p 0 vfl_decision_tree
./semi-party.x -F -N 3 -I -p 1 vfl_decision_tree
./semi-party.x -F -N 3 -I -p 2 vfl_decision_tree
```

After running the commands, there should be the following output in the terminals

```
Player 0 is running on machine 127.0.0.1
Player 1 is running on machine 127.0.0.1
Player 2 is running on machine 127.0.0.1
Reading MAC keys from Player-Data/3-128-128//Player-MAC-Keys-p-P0
Tape has unknown usage
Using single-threaded receiving
Using indirect communication.
Start listening on thread 139939100210944
Party 0 is listening on port 18000 for external client connections.
Listening for client connections on base port 18000
```

Then, run the pivot parties on the other three terminals by the following commands.

```
./Pivot --client-id 0 --client-num 3 --class-num 2 --algorithm-type 0 --tree-type 0 --solution-type 0 --optimization-type 1 --network-file /opt/pivot/data/networks/Parties.txt --data-file /opt/pivot/data/bank_marketing_data/client_0.txt --logger-file /opt/pivot/log/bank_marketing_data --max-bins 16 --max-depth 3 --num-trees 1
./Pivot --client-id 1 --client-num 3 --class-num 2 --algorithm-type 0 --tree-type 0 --solution-type 0 --optimization-type 1 --network-file /opt/pivot/data/networks/Parties.txt --data-file /opt/pivot/data/bank_marketing_data/client_1.txt --logger-file /opt/pivot/log/bank_marketing_data --max-bins 16 --max-depth 3 --num-trees 1
./Pivot --client-id 2 --client-num 3 --class-num 2 --algorithm-type 0 --tree-type 0 --solution-type 0 --optimization-type 1 --network-file /opt/pivot/data/networks/Parties.txt --data-file /opt/pivot/data/bank_marketing_data/client_2.txt --logger-file /opt/pivot/log/bank_marketing_data --max-bins 16 --max-depth 3 --num-trees 1
```

The parties will run the program, and the logs can be found under `/opt/pivot/log/`.