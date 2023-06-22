# FROM ubuntu:16.04
# FROM debian:stable
FROM ubuntu:18.04

WORKDIR /app
RUN apt-get update
RUN apt-get install git -y
RUN git clone https://github.com/xiaoyeli/superlu_dist.git
WORKDIR superlu_dist
RUN git fetch
RUN git pull
RUN git checkout gpu_trisolve_new
RUN bash config_cleanlinux.sh 
