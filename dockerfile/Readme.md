# Using the docker file


This dockerfile is a somewhat simpler way of running the parsing and classification process. The below commands are for Linux users.

## Build

build the docker file using by entering the following command into the command line interface from the base of the repo.

docker build ./dockerfile/ -t ocod/parse_process:latest


## Run

To run the script enter to below from the base of the repo

docker run --rm -it -v $(pwd):/app ocod/parse_process:latest ./app/full_ocod_parse_process.py ./app/empty_homes_data/
