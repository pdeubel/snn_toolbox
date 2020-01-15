FROM nestsim/nest:latest

RUN apt-get update -y
RUN apt-get install -y python3-pip

RUN python3 -m pip install future keras pyNN numpy==1.17.4 tensorflow